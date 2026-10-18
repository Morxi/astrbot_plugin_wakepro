import asyncio
import math

from astrbot.api import logger

from ..config import PluginConfig
from ..jev import JevClient
from ..model import StepName, StepResult, WakeContext
from .base import BaseStep


class JevStep(BaseStep):
    """Use group context and Jev to detect messages addressed to the bot."""

    name = StepName.JEV

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.cfg = config.jev
        self.client = JevClient(self.cfg)

    async def handle(self, ctx: WakeContext) -> StepResult:
        """Classify the current group message with Jev when configured.

        Args:
            ctx: Current WakePro message context.

        Returns:
            A wake result when Jev reaches the configured threshold.
        """
        if (
            not self.cfg.jev_api_key
            or ctx.cmd
            or ctx.debounce_follow_up
            or ctx.event.is_at_or_wake_command
            or not ctx.group
            or not ctx.plain
        ):
            return StepResult()
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
            "text": ctx.plain,
            "time": ctx.now,
            "is_bot": False,
            "bot_id": str(ctx.bid),
        }
        history = [
            item for item in ctx.group.history if item.get("message_id") != message_id
        ][-history_messages:]
        names = [*self.config.wake_prefix, *self.config.mention.names]
        names.extend(self.config.mention.admin_names)
        try:
            probability = await self.client.judge(
                history, candidate, str(ctx.bid), names
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                "Jev wake judgment failed; skipping semantic wake: %s", error
            )
            return StepResult()
        if probability is not None and probability >= threshold:
            return StepResult(
                wake=True, prolong=True, msg=f"Jev 语义唤醒({probability:.2%})"
            )
        return StepResult(msg="Jev 未达到唤醒阈值")

    async def terminate(self) -> None:
        """Close the Jev HTTP client session."""
        await self.client.close()
