import asyncio
import math
import re
import statistics

from astrbot.api import logger

from ..config import PluginConfig
from ..jev import JevClient, render_chain, render_history_row
from ..model import StepName, StepResult, WakeContext
from .base import BaseStep


def _number(value: object, default: float) -> float:
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    return default


def calibrate(probability: float, center: float, sharpness: float) -> float:
    """Re-center and stretch a probability in logit space.

    ``center`` maps to 0.5 and ``sharpness`` scales the distance from it, which
    undoes the compressed output of small or high-temperature Jev models. The
    defaults (0.5, 1) return the probability unchanged.
    """
    if center == 0.5 and sharpness == 1:
        return probability
    eps = 1e-6
    p = min(max(probability, eps), 1 - eps)
    c = min(max(center, eps), 1 - eps)
    z = (math.log(p / (1 - p)) - math.log(c / (1 - c))) * sharpness
    return 1 / (1 + math.exp(-max(min(z, 50), -50)))


def _is_name(value: object) -> bool:
    """Reject wake prefixes such as ``/`` that are not names the bot is called."""
    return isinstance(value, str) and bool(re.search(r"\w", value))


class JevStep(BaseStep):
    """Use group context and Jev to detect messages addressed to the bot."""

    name = StepName.JEV

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.cfg = config.jev
        self.client = JevClient(self.cfg)
        self._nicknames: dict[tuple[str, str], list[str]] = {}

    @property
    def available(self) -> bool:
        return bool(self.cfg.jev_api_url) and bool(
            self.cfg.jev_api_key or self.cfg.allow_empty_key
        )

    async def initialize(self) -> None:
        """Check that the configured Jev model is served."""
        if self.available and self.config.pipeline.is_enabled_step(self.name):
            await self.client.check_model()

    async def _detect_nicknames(self, ctx: WakeContext) -> list[str]:
        """Fetch the bot's group card and nickname from OneBot, cached per group."""
        if self.cfg.auto_nickname is False:
            return []
        key = (str(ctx.bid), str(ctx.gid))
        if key in self._nicknames:
            return self._nicknames[key]
        names: list[str] = []
        bot = getattr(ctx.event, "bot", None)
        call = getattr(bot, "call_action", None)
        if call is not None:
            try:
                info = await asyncio.wait_for(
                    call(
                        "get_group_member_info",
                        group_id=int(ctx.gid),
                        user_id=int(ctx.bid),
                        no_cache=False,
                    ),
                    5,
                )
                if isinstance(info, dict):
                    names += [info.get("card") or "", info.get("nickname") or ""]
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.debug("Jev failed to fetch bot nickname: %s", error)
        names = [name for name in dict.fromkeys(names) if _is_name(name)]
        if names or call is None:
            self._nicknames[key] = names
        return names

    async def _bot_names(self, ctx: WakeContext) -> list[str]:
        names = [
            *(self.cfg.bot_nicknames or []),
            *(await self._detect_nicknames(ctx)),
            *self.config.mention.names,
            *self.config.mention.admin_names,
            *self.config.wake_prefix,
            *(self.cfg.bot_aliases or []),
        ]
        return [name for name in dict.fromkeys(names) if _is_name(name)]

    def _astrbot_history_enabled(self, ctx: WakeContext) -> bool:
        source = str(self.cfg.history_source or "auto").split("(", 1)[0].strip()
        if source == "plugin":
            return False
        if source == "astrbot":
            return True
        try:
            settings = self.config.context.get_config(umo=ctx.umo).get(
                "provider_ltm_settings", {}
            )
        except Exception:
            return False
        return bool(settings.get("group_message_history_enable"))

    async def _history(self, ctx: WakeContext, message_id: str, limit: int) -> list:
        """Read context from AstrBot's persisted group history, else plugin memory."""
        if self._astrbot_history_enabled(ctx):
            manager = getattr(self.config.context, "message_history_manager", None)
            if manager is not None:
                try:
                    rows = await manager.get(
                        platform_id=ctx.event.get_platform_id(),
                        user_id=ctx.umo,
                        page=1,
                        page_size=limit + 1,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning("Jev failed to read AstrBot history: %s", error)
                else:
                    current = ctx.event.get_extra(
                        "_current_platform_message_history_id"
                    )
                    records = [
                        record
                        for row in rows
                        if getattr(row, "id", None) != current
                        and (record := render_history_row(row)) is not None
                    ]
                    if records:
                        return records[-limit:]
        return [
            item for item in ctx.group.history if item.get("message_id") != message_id
        ][-limit:]

    async def handle(self, ctx: WakeContext) -> StepResult:
        """Classify the current group message with Jev when configured.

        Args:
            ctx: Current WakePro message context.

        Returns:
            A wake result when Jev reaches the configured threshold, or when the
            adaptive mode sees a score clearly above the group's baseline.
        """
        if (
            not self.available
            or ctx.cmd
            or ctx.debounce_follow_up
            or ctx.event.is_at_or_wake_command
            or not ctx.group
            or not ctx.plain
        ):
            return StepResult()
        if len(ctx.plain.strip()) < int(_number(self.cfg.min_chars, 0)):
            return StepResult(msg="Jev 跳过过短消息")
        threshold = self.cfg.jev_threshold
        history_messages = self.cfg.history_messages
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 < threshold <= 1
            or type(history_messages) is not int
            or history_messages < 1
        ):
            logger.warning("Invalid Jev configuration; skipping semantic wake")
            return StepResult()

        message_id = str(getattr(ctx.event.message_obj, "message_id", "") or "")
        if not message_id:
            message_id = f"{ctx.uid}:{ctx.now}"
        candidate = {
            "message_id": message_id,
            "sender_id": str(ctx.uid),
            "sender_name": ctx.event.get_sender_name() or "",
            "text": render_chain(ctx.chain) or ctx.plain,
            "time": ctx.now,
            "is_bot": False,
            "bot_id": str(ctx.bid),
        }
        history = await self._history(ctx, message_id, history_messages)
        names = await self._bot_names(ctx)
        try:
            raw = await self.client.judge(history, candidate, str(ctx.bid), names)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                "Jev wake judgment failed; skipping semantic wake: %s", error
            )
            return StepResult()
        if raw is None:
            return StepResult()

        probability = calibrate(
            raw,
            _number(self.cfg.calibration_center, 0.5),
            _number(self.cfg.calibration_sharpness, 1.0),
        )
        wake = probability >= threshold
        reason = f"{probability:.2%}≥{threshold:.2%}"

        # Adaptive mode: most group messages are not addressed to the bot, so the
        # median of recent raw scores approximates this group's "not for me"
        # baseline. A score well above it is treated as addressed to the bot.
        scores = ctx.group.jev_scores
        margin = _number(self.cfg.adaptive_margin, 0)
        window = max(int(_number(self.cfg.adaptive_window, 30)), 1)
        min_samples = max(int(_number(self.cfg.adaptive_min_samples, 10)), 1)
        recent = list(scores)[-window:]
        baseline = statistics.median(recent) if recent else None
        if (
            not wake
            and margin > 0
            and baseline is not None
            and len(recent) >= min_samples
            and raw - baseline >= margin
        ):
            wake = True
            reason = f"原始 {raw:.2%} 高于群基线 {baseline:.2%} + {margin:.2%}"
        scores.append(raw)

        if self.cfg.debug_log:
            logger.info(
                "Jev decision | message=%s | raw=%.4f | calibrated=%.4f | "
                "threshold=%.4f | baseline=%s (%d samples) | wake=%s",
                message_id,
                raw,
                probability,
                threshold,
                f"{baseline:.4f}" if baseline is not None else "-",
                len(recent),
                wake,
            )
        if wake:
            return StepResult(wake=True, prolong=True, msg=f"Jev 语义唤醒({reason})")
        return StepResult(msg=f"Jev 未达到唤醒阈值({probability:.2%})")

    async def terminate(self) -> None:
        """Close the Jev HTTP client session."""
        await self.client.close()
