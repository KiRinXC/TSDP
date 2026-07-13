"""TEESlice 使用的 CIFAR ResNet18 与私有 slice 结构。"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import models as tv_models


PUBLISHED_C100_R18_KEEP_FLAGS = (
    (True, False),
    (True, True, True),
    (True, False, False, False),
    (True, True, False, True),
    (True, False, False, False),
    (True, True, False, True),
    (True, False, False, False),
    (True, True, True, True),
)
PUBLISHED_C100_R18_TASK_PARAMS = 711_524
PUBLISHED_C100_R18_TASK_FLOPS = 29_868_032


def _load_official_state(weight_path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(weight_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 ResNet18 官方预训练权重：{path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "conv1.weight" not in state:
        raise ValueError(f"无法识别 ResNet18 官方权重：{path}")
    return state


class CifarResNet18(nn.Module):
    """将官方 ResNet18 改为 TEESlice 使用的 CIFAR 特征分辨率。"""

    def __init__(self, num_classes: int, weight_path: str | Path):
        super().__init__()
        base = tv_models.resnet18(weights=None, num_classes=num_classes)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.last_linear = base.fc
        self.load_public_weights(weight_path)

    def load_public_weights(self, weight_path: str | Path) -> None:
        official = _load_official_state(weight_path)
        state = self.state_dict()
        for name in state:
            if name.startswith("last_linear."):
                continue
            source_name = name
            if source_name.endswith("num_batches_tracked") and source_name not in official:
                continue
            source = official[source_name]
            if source_name == "conv1.weight":
                source = source[:, :, 2:5, 2:5]
            if source.shape != state[name].shape:
                raise ValueError(
                    f"官方权重 {source_name} 形状 {tuple(source.shape)} 与 CIFAR 模型 "
                    f"{tuple(state[name].shape)} 不一致。"
                )
            state[name] = source
        self.load_state_dict(state, strict=True)

    def forward_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.relu(self.bn1(self.conv1(inputs)))
        block_outputs: list[torch.Tensor] = []
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                x = block(x)
                block_outputs.append(x)
        return x, block_outputs

    def forward(self, inputs: torch.Tensor, return_features: bool = False):
        x, block_outputs = self.forward_features(inputs)
        logits = self.last_linear(self.avgpool(x).flatten(1))
        if return_features:
            return logits, block_outputs
        return logits


class SliceProxy(nn.Module):
    """把一个较早的 block 输出投影到当前 block 的形状。"""

    def __init__(self, in_channels: int, out_channels: int, spatial_stride: int):
        super().__init__()
        if spatial_stride < 1:
            raise ValueError("proxy spatial_stride 必须大于等于 1。")
        self.spatial_stride = spatial_stride
        self.pool = (
            nn.MaxPool2d(
                kernel_size=spatial_stride + 1,
                stride=spatial_stride,
                padding=spatial_stride // 2,
            )
            if spatial_stride > 1
            else nn.Identity()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(self.pool(inputs)))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def flops(self, source_shape: tuple[int, int, int], target_shape: tuple[int, int, int]) -> int:
        in_channels = source_shape[0]
        out_channels, height, width = target_shape
        spatial = height * width
        return spatial * in_channels * out_channels + spatial * out_channels * 2


class SliceBlock(nn.Module):
    """一个公开 BasicBlock 与若干私有 skip proxy 的混合。"""

    def __init__(
        self,
        main: nn.Module,
        source_shapes: Sequence[tuple[int, int, int]],
        target_shape: tuple[int, int, int],
    ):
        super().__init__()
        self.main = main
        self.source_shapes = tuple(source_shapes)
        self.target_shape = target_shape
        proxies = []
        for source_shape in self.source_shapes:
            if source_shape[1] % target_shape[1] != 0:
                raise ValueError("proxy 输入输出空间尺寸不能整除。")
            proxies.append(
                SliceProxy(
                    in_channels=source_shape[0],
                    out_channels=target_shape[0],
                    spatial_stride=source_shape[1] // target_shape[1],
                )
            )
        self.proxies = nn.ModuleList(proxies)
        alpha = torch.full((len(proxies) + 1,), -2.0)
        alpha[0] = 2.0
        self.alpha = nn.Parameter(alpha)
        self.register_buffer("keep_mask", torch.ones(len(proxies) + 1, dtype=torch.bool))

    def active_alphas(self) -> torch.Tensor:
        if not bool(self.keep_mask[0]):
            raise ValueError("TEESlice universal main path 不能被删除。")
        masked = self.alpha.masked_fill(~self.keep_mask, torch.finfo(self.alpha.dtype).min)
        return functional.softmax(masked, dim=0)

    def forward(self, ends: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(ends) != len(self.proxies):
            raise ValueError("SliceBlock 输入历史数量与 proxy 数量不一致。")
        alphas = self.active_alphas()
        output = alphas[0] * self.main(ends[0])
        for index, proxy in enumerate(self.proxies, start=1):
            if bool(self.keep_mask[index]):
                output = output + alphas[index] * proxy(ends[index - 1])
        return functional.relu(output, inplace=False)

    def proxy_scores(self) -> list[float]:
        alphas = self.active_alphas().detach().cpu()
        return [float(alphas[index]) for index in range(1, len(alphas))]

    def proxy_parameter_counts(self) -> list[int]:
        return [proxy.parameter_count() for proxy in self.proxies]

    def proxy_flops(self) -> list[int]:
        return [
            proxy.flops(source_shape, self.target_shape)
            for proxy, source_shape in zip(self.proxies, self.source_shapes, strict=True)
        ]


class TEESliceResNet18(nn.Module):
    """TEESlice 的 ResNet18 public-backbone/private-slice defended victim。"""

    def __init__(
        self,
        num_classes: int,
        weight_path: str | Path,
        max_skip: int = 3,
        image_size: int = 32,
    ):
        super().__init__()
        if max_skip <= 0:
            raise ValueError("max_skip 必须大于 0。")
        if image_size <= 0:
            raise ValueError("image_size 必须大于 0。")

        public = CifarResNet18(num_classes=num_classes, weight_path=weight_path)
        self.num_classes = num_classes
        self.max_skip = max_skip
        self.image_size = image_size
        self.conv1 = public.conv1
        self.bn1 = public.bn1
        self.relu = public.relu
        self.avgpool = public.avgpool

        public_blocks = [block for stage in (public.layer1, public.layer2, public.layer3, public.layer4) for block in stage]
        current_shape = (64, image_size, image_size)
        history_shapes: deque[tuple[int, int, int]] = deque(maxlen=max_skip)
        history_shapes.appendleft(current_shape)
        blocks: list[SliceBlock] = []
        self._main_flops: list[int] = []
        for main in public_blocks:
            stride = int(main.conv1.stride[0])
            target_shape = (int(main.conv2.out_channels), current_shape[1] // stride, current_shape[2] // stride)
            blocks.append(SliceBlock(copy.deepcopy(main), tuple(history_shapes), target_shape))
            self._main_flops.append(self._basic_block_flops(main, current_shape, target_shape))
            current_shape = target_shape
            history_shapes.appendleft(current_shape)
        self.blocks = nn.ModuleList(blocks)
        self.last_linear = nn.Linear(current_shape[0], num_classes)
        self.freeze_public()

    @staticmethod
    def _basic_block_flops(
        block: nn.Module,
        source_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
    ) -> int:
        spatial = target_shape[1] * target_shape[2]
        in_channels = source_shape[0]
        out_channels = target_shape[0]
        flops = spatial * in_channels * out_channels * 9
        flops += spatial * out_channels * 2
        flops += spatial * out_channels * out_channels * 9
        flops += spatial * out_channels * 2
        if block.downsample is not None:
            flops += spatial * in_channels * out_channels
            flops += spatial * out_channels * 2
        return flops

    def freeze_public(self) -> None:
        for parameter in self.conv1.parameters():
            parameter.requires_grad = False
        for parameter in self.bn1.parameters():
            parameter.requires_grad = False
        for block in self.blocks:
            for parameter in block.main.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor, return_features: bool = False):
        x = self.relu(self.bn1(self.conv1(inputs)))
        ends: deque[torch.Tensor] = deque(maxlen=self.max_skip)
        ends.appendleft(x)
        supervised_features: list[torch.Tensor] = []
        for block in self.blocks:
            block_inputs = tuple(ends)
            if return_features:
                supervised_features.append(block(tuple(value.detach() for value in block_inputs)))
            x = block(block_inputs)
            ends.appendleft(x)
        logits = self.last_linear(self.avgpool(x).flatten(1))
        if return_features:
            return logits, supervised_features
        return logits

    def private_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for block in self.blocks:
            parameters.append(block.alpha)
            parameters.extend(block.proxies.parameters())
        parameters.extend(self.last_linear.parameters())
        return [parameter for parameter in parameters if parameter.requires_grad]

    def get_keep_flags(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(tuple(bool(value) for value in block.keep_mask.tolist()) for block in self.blocks)

    def set_keep_flags(self, flags: Sequence[Sequence[bool]]) -> None:
        if len(flags) != len(self.blocks):
            raise ValueError("keep flag block 数量不一致。")
        for block, block_flags in zip(self.blocks, flags, strict=True):
            if len(block_flags) != len(block.keep_mask):
                raise ValueError("keep flag path 数量不一致。")
            if not bool(block_flags[0]):
                raise ValueError("TEESlice universal main path 必须保留。")
            block.keep_mask.copy_(torch.tensor(block_flags, dtype=torch.bool, device=block.keep_mask.device))

    def active_proxy_count(self) -> int:
        return sum(int(block.keep_mask[1:].sum().item()) for block in self.blocks)

    def _proxy_rank(self, active_only: bool) -> list[tuple[float, int, int]]:
        ranked: list[tuple[float, int, int]] = []
        for block_index, block in enumerate(self.blocks):
            scores = block.proxy_scores()
            for proxy_index, score in enumerate(scores):
                if active_only and not bool(block.keep_mask[proxy_index + 1]):
                    continue
                ranked.append((score, block_index, proxy_index))
        return sorted(ranked, key=lambda item: (item[0], item[1], item[2]))

    def initial_prune(self, threshold: float, minimum_fraction: float) -> list[tuple[int, int]]:
        if not 0.0 <= minimum_fraction < 1.0:
            raise ValueError("minimum_fraction 必须位于 [0, 1)。")
        ranked = self._proxy_rank(active_only=False)
        if threshold > max(score for score, _, _ in ranked):
            threshold_count = int(len(ranked) * 0.8)
            remove = {
                (block_index, proxy_index)
                for _, block_index, proxy_index in ranked[:threshold_count]
            }
        else:
            remove = {
                (block_index, proxy_index)
                for score, block_index, proxy_index in ranked
                if score <= threshold
            }
        minimum = int(len(ranked) * minimum_fraction)
        remove.update((block_index, proxy_index) for _, block_index, proxy_index in ranked[:minimum])
        if len(remove) >= len(ranked):
            remove.remove((ranked[-1][1], ranked[-1][2]))
        for block_index, proxy_index in sorted(remove):
            self.blocks[block_index].keep_mask[proxy_index + 1] = False
        return sorted(remove)

    def iterative_prune(self, fraction_of_all: float) -> list[tuple[int, int]]:
        if fraction_of_all <= 0.0:
            raise ValueError("fraction_of_all 必须大于 0。")
        all_count = sum(len(block.proxies) for block in self.blocks)
        active = self._proxy_rank(active_only=True)
        if len(active) <= 1:
            return []
        remove_count = max(1, int(all_count * fraction_of_all))
        remove_count = min(remove_count, len(active) - 1)
        removed = [(block_index, proxy_index) for _, block_index, proxy_index in active[:remove_count]]
        for block_index, proxy_index in removed:
            self.blocks[block_index].keep_mask[proxy_index + 1] = False
        return removed

    def cost_summary(self) -> dict[str, int | float]:
        public_params = sum(parameter.numel() for parameter in self.conv1.parameters())
        public_params += sum(parameter.numel() for parameter in self.bn1.parameters())
        proxy_params = 0
        proxy_flops = 0
        active_alpha_count = 0
        for block in self.blocks:
            public_params += sum(parameter.numel() for parameter in block.main.parameters())
            active_alpha_count += int(block.keep_mask.sum().item())
            for index, (param_count, flop_count) in enumerate(
                zip(block.proxy_parameter_counts(), block.proxy_flops(), strict=True),
                start=1,
            ):
                if bool(block.keep_mask[index]):
                    proxy_params += param_count
                    proxy_flops += flop_count
        head_params = sum(parameter.numel() for parameter in self.last_linear.parameters())
        paper_private_params = proxy_params + head_params
        actual_private_params = paper_private_params + active_alpha_count
        total_params = public_params + actual_private_params
        private_bn_buffer_count = sum(
            buffer.numel()
            for name, buffer in self.state_dict().items()
            if (
                name.endswith("running_mean")
                or name.endswith("running_var")
                or name.endswith("num_batches_tracked")
            )
        )

        stem_spatial = self.image_size * self.image_size
        public_flops = stem_spatial * 3 * 64 * 9 + stem_spatial * 64 * 2
        public_flops += sum(self._main_flops)
        head_flops = self.last_linear.in_features * self.last_linear.out_features + self.last_linear.out_features
        private_flops = proxy_flops + head_flops
        total_flops = public_flops + private_flops
        return {
            "active_proxy_count": self.active_proxy_count(),
            "active_alpha_count": active_alpha_count,
            "public_param_count": public_params,
            "paper_private_param_count": paper_private_params,
            "private_param_count": actual_private_params,
            "private_bn_buffer_count": private_bn_buffer_count,
            "total_param_count": total_params,
            "private_param_ratio": actual_private_params / total_params,
            "paper_private_flops": proxy_flops,
            "private_flops": private_flops,
            "public_flops": public_flops,
            "total_flops": total_flops,
            "private_flops_ratio": private_flops / total_flops,
        }

    def expected_complexity(self) -> torch.Tensor:
        """复现作者 full-model 阶段用于学习路径 alpha 的复杂度项。"""
        global_flops = float(
            self.image_size * self.image_size * 3 * 64 * 9
            + self.image_size * self.image_size * 64 * 2
            + sum(self._main_flops)
        )
        alphas_list = [block.active_alphas() for block in self.blocks]
        proxy_complexities = [
            [float(value) / global_flops for value in block.proxy_flops()]
            for block in self.blocks
        ]
        incoming_clear = [torch.prod(1.0 - alphas) for alphas in alphas_list]
        outgoing: list[list[torch.Tensor]] = [[] for _ in alphas_list]
        for block_index, alphas in enumerate(alphas_list):
            for path_index in range(len(alphas)):
                source_index = block_index if path_index <= 1 else block_index - (path_index - 1)
                outgoing[source_index].append(alphas[path_index])
        outgoing_clear = [torch.prod(1.0 - torch.stack(values)) for values in outgoing]

        complexity = torch.zeros((), device=self.blocks[0].alpha.device)
        for block_index, (alphas, costs) in enumerate(zip(alphas_list, proxy_complexities, strict=True)):
            for proxy_index, cost in enumerate(costs, start=1):
                probability = alphas[proxy_index]
                p_in = (
                    torch.ones((), device=probability.device)
                    if block_index == 0
                    else incoming_clear[block_index - 1]
                )
                p_out = (
                    torch.ones((), device=probability.device)
                    if block_index == len(outgoing_clear) - 1
                    else outgoing_clear[block_index + 1]
                )
                complexity = complexity + cost * (probability - p_in - p_out)
        return complexity

    @staticmethod
    def _is_public_backbone_name(name: str) -> bool:
        return name.split(".", 1)[0] in {"conv1", "bn1"} or ".main." in name

    def public_state_dict(self) -> dict[str, torch.Tensor]:
        """返回攻击者已知且训练期间不可变的 universal backbone 状态。"""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
            if self._is_public_backbone_name(name)
            and not (
                name.endswith("running_mean")
                or name.endswith("running_var")
                or name.endswith("num_batches_tracked")
            )
        }

    def private_bn_state_dict(self) -> dict[str, torch.Tensor]:
        """返回作者实现会随任务数据适配并保存在私有 checkpoint 中的全部 BN buffer。"""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
            if (
                name.endswith("running_mean")
                or name.endswith("running_var")
                or name.endswith("num_batches_tracked")
            )
        }


def cifar_resnet18(num_classes: int, weight_path: str | Path) -> CifarResNet18:
    return CifarResNet18(num_classes=num_classes, weight_path=weight_path)


def teeslice_r18(
    num_classes: int,
    weight_path: str | Path,
    max_skip: int = 3,
    image_size: int = 32,
) -> TEESliceResNet18:
    return TEESliceResNet18(
        num_classes=num_classes,
        weight_path=weight_path,
        max_skip=max_skip,
        image_size=image_size,
    )
