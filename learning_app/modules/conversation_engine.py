"""
Conversation Engine Module for RAG Chat.

Manages conversation state and history:
- Track user and assistant messages
- Format messages for display
- Manage conversation history
- Provide conversation statistics

Keeps implementation simple for Phase 3 basics.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """Represents a single message in the conversation."""

    role: str              # "user" or "assistant"
    content: str          # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Message':
        """Create Message from dictionary."""
        return Message(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            metadata=data.get("metadata", {})
        )


class ConversationEngine:
    """
    Manages conversation state for RAG chat.

    Responsibilities:
    - Store message history
    - Format messages for display
    - Provide conversation context to RAG pipeline
    - Track conversation metadata

    Educational focus: Shows how chat systems maintain state
    across multiple turns.
    """

    def __init__(self, max_history_turns: int = 10):
        """
        Initialize conversation engine.

        Args:
            max_history_turns: Maximum number of conversation turns to keep
                              (1 turn = 1 user message + 1 assistant response)
        """
        self.max_history_turns = max_history_turns
        self.messages: List[Message] = []
        self.conversation_id = self._generate_id()
        self.started_at = datetime.now()

    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> Message:
        """
        Add user message to conversation.

        Args:
            content: User message text
            metadata: Optional metadata

        Returns:
            Created Message object
        """
        message = Message(
            role="user",
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self._prune_history()
        return message

    def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Message:
        """
        Add assistant message to conversation.

        Args:
            content: Assistant response text
            metadata: Optional metadata (sources, tokens, etc.)

        Returns:
            Created Message object
        """
        message = Message(
            role="assistant",
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self._prune_history()
        return message

    def get_history_for_rag(self) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for RAG pipeline.

        Returns:
            List of {role, content} dictionaries
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def get_formatted_history(self) -> str:
        """
        Get conversation history formatted for display.

        Returns:
            Markdown-formatted conversation string
        """
        if not self.messages:
            return "_No messages yet_"

        lines = []
        for msg in self.messages:
            prefix = "**You:**" if msg.role == "user" else "**Assistant:**"
            lines.append(f"{prefix} {msg.content}\n")

        return "\n".join(lines)

    def get_history_html(self) -> str:
        """
        Get conversation history as HTML.

        Returns:
            HTML string for display in Gradio
        """
        if not self.messages:
            return "<div style='opacity: 0.6; font-style: italic;'>No messages yet. Start a conversation!</div>"

        html_parts = []
        for msg in self.messages:
            if msg.role == "user":
                html_parts.append(f"""
                <div style="margin: 10px 0; padding: 12px; background: rgba(59, 130, 246, 0.1); border-left: 4px solid rgba(59, 130, 246, 0.6); border-radius: 4px;">
                    <div style="font-weight: bold; opacity: 0.9; margin-bottom: 5px;">👤 You</div>
                    <div>{self._escape_html(msg.content)}</div>
                </div>
                """)
            else:
                html_parts.append(f"""
                <div style="margin: 10px 0; padding: 12px; background: var(--block-background-fill); border-left: 4px solid var(--border-color-primary); border-radius: 4px;">
                    <div style="font-weight: bold; opacity: 0.9; margin-bottom: 5px;">🤖 Assistant</div>
                    <div style="white-space: pre-wrap;">{self._escape_html(msg.content)}</div>
                    {self._format_sources(msg.metadata.get('sources', []))}
                </div>
                """)

        return "".join(html_parts)

    def _format_sources(self, sources: List[Dict]) -> str:
        """Format source citations for message."""
        if not sources:
            return ""

        html = "<div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-color-primary);'>"
        html += "<div style='font-size: 0.85em; opacity: 0.7; margin-bottom: 5px;'>📚 Sources:</div>"

        for src in sources:
            score = src.get('score', 0.0)
            text_preview = src.get('text', '')[:100]
            html += f"""
            <div style='font-size: 0.8em; opacity: 0.7; margin-left: 15px; margin-bottom: 3px;'>
                • [{src.get('id', '?')}] (Score: {score:.3f}) {text_preview}...
            </div>
            """

        html += "</div>"
        return html

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))

    def clear(self):
        """Clear conversation history."""
        self.messages = []
        self.conversation_id = self._generate_id()
        self.started_at = datetime.now()

    def get_stats(self) -> Dict:
        """
        Get conversation statistics.

        Returns:
            Dictionary with stats
        """
        user_messages = sum(1 for msg in self.messages if msg.role == "user")
        assistant_messages = sum(1 for msg in self.messages if msg.role == "assistant")

        return {
            "conversation_id": self.conversation_id,
            "started_at": self.started_at.isoformat(),
            "total_messages": len(self.messages),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "turns": min(user_messages, assistant_messages),
            "max_history_turns": self.max_history_turns
        }

    def get_stats_html(self) -> str:
        """
        Get conversation statistics as HTML.

        Returns:
            HTML string with stats
        """
        stats = self.get_stats()

        html = f"""
        <div style="font-family: sans-serif; padding: 12px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 8px; font-size: 0.9em;">
            <strong>📊 Conversation Stats</strong>
            <div style="margin-top: 8px; opacity: 0.8;">
                <div>💬 Total messages: {stats['total_messages']}</div>
                <div>🔄 Turns: {stats['turns']}</div>
                <div>👤 Your messages: {stats['user_messages']}</div>
                <div>🤖 Assistant replies: {stats['assistant_messages']}</div>
                <div>📝 History limit: {stats['max_history_turns']} turns</div>
            </div>
        </div>
        """

        return html

    def _prune_history(self):
        """
        Prune conversation history to stay within limits.

        Strategy: Keep most recent turns (user + assistant pairs).
        """
        if len(self.messages) <= (self.max_history_turns * 2):
            return

        # Calculate how many messages to keep
        # Keep pairs of (user, assistant) messages
        max_messages = self.max_history_turns * 2

        # Keep most recent messages
        self.messages = self.messages[-max_messages:]

    def _generate_id(self) -> str:
        """Generate unique conversation ID."""
        import random
        import string
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(12))

    def export_to_dict(self) -> Dict:
        """
        Export conversation to dictionary for saving.

        Returns:
            Dictionary with full conversation data
        """
        return {
            "conversation_id": self.conversation_id,
            "started_at": self.started_at.isoformat(),
            "max_history_turns": self.max_history_turns,
            "messages": [msg.to_dict() for msg in self.messages],
            "stats": self.get_stats()
        }

    @staticmethod
    def load_from_dict(data: Dict) -> 'ConversationEngine':
        """
        Load conversation from dictionary.

        Args:
            data: Exported conversation data

        Returns:
            ConversationEngine instance
        """
        engine = ConversationEngine(
            max_history_turns=data.get("max_history_turns", 10)
        )
        engine.conversation_id = data.get("conversation_id", engine.conversation_id)
        engine.started_at = datetime.fromisoformat(
            data.get("started_at", datetime.now().isoformat())
        )
        engine.messages = [
            Message.from_dict(msg_data)
            for msg_data in data.get("messages", [])
        ]
        return engine

    def get_last_user_message(self) -> Optional[str]:
        """Get last user message content."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get last assistant message content."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def has_messages(self) -> bool:
        """Check if conversation has any messages."""
        return len(self.messages) > 0
