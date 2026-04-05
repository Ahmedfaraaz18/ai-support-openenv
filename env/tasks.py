from typing import Dict, List, Optional
from pydantic import BaseModel


class Ticket(BaseModel):
    ticket_id: int
    level: str
    message: str
    user_history: List[str]
    current_status: str
    urgency_hint: str
    expected_category: str
    expected_priority: str
    expected_keywords: List[str]


TICKETS: List[Ticket] = [
    # EASY (5)
    Ticket(
        ticket_id=1,
        level="easy",
        message="I was charged twice for my last invoice, please fix it.",
        user_history=["Last month payment double charged", "Reached out via chat"],
        current_status="open",
        urgency_hint="high",
        expected_category="billing",
        expected_priority="high",
        expected_keywords=["refund", "invoice", "charge", "resolved"],
    ),
    Ticket(
        ticket_id=2,
        level="easy",
        message="My password reset link is not working.",
        user_history=["created account yesterday"],
        current_status="open",
        urgency_hint="medium",
        expected_category="account",
        expected_priority="medium",
        expected_keywords=["reset", "password", "link", "assistance"],
    ),
    Ticket(
        ticket_id=3,
        level="easy",
        message="My app crashes when I click send on messages.",
        user_history=["never happened before"],
        current_status="open",
        urgency_hint="medium",
        expected_category="technical",
        expected_priority="medium",
        expected_keywords=["crash", "fix", "bug", "steps"],
    ),
    Ticket(
        ticket_id=4,
        level="easy",
        message="Please close my account immediately.",
        user_history=["cancelled subscription"],
        current_status="open",
        urgency_hint="low",
        expected_category="account",
        expected_priority="low",
        expected_keywords=["close", "account", "confirm", "completed"],
    ),
    Ticket(
        ticket_id=5,
        level="easy",
        message="Can you explain why my last invoice is higher than normal?",
        user_history=["previous invoices usually lower"],
        current_status="open",
        urgency_hint="low",
        expected_category="billing",
        expected_priority="low",
        expected_keywords=["invoice", "explanation", "breakdown", "clarify"],
    ),
    # MEDIUM (5)
    Ticket(
        ticket_id=6,
        level="medium",
        message="I have trouble upgrading and also saw an extra charge.",
        user_history=["tried premium upgrade", "reviewed billing"],
        current_status="open",
        urgency_hint="high",
        expected_category="billing",
        expected_priority="high",
        expected_keywords=["upgrade", "charge", "billing", "plan"],
    ),
    Ticket(
        ticket_id=7,
        level="medium",
        message="Sometimes app won't open after update and I also need to change my email.",
        user_history=["updated yesterday", "profile replacement"],
        current_status="open",
        urgency_hint="medium",
        expected_category="technical",
        expected_priority="medium",
        expected_keywords=["update", "app", "email", "access"],
    ),
    Ticket(
        ticket_id=8,
        level="medium",
        message="I cannot access my old tickets and there's a wrong billing amount.",
        user_history=["support history lost"],
        current_status="open",
        urgency_hint="high",
        expected_category="billing",
        expected_priority="high",
        expected_keywords=["access", "history", "billing", "fix"],
    ),
    Ticket(
        ticket_id=9,
        level="medium",
        message="How can I reduce spam notifications and also update payment details?",
        user_history=["frequent alerts"],
        current_status="open",
        urgency_hint="low",
        expected_category="account",
        expected_priority="medium",
        expected_keywords=["notifications", "settings", "payment", "update"],
    ),
    Ticket(
        ticket_id=10,
        level="medium",
        message="I get timeout errors when uploading files; also billing limit seems odd.",
        user_history=["upload succeeded yesterday"],
        current_status="open",
        urgency_hint="high",
        expected_category="technical",
        expected_priority="high",
        expected_keywords=["timeout", "upload", "limit", "billing"],
    ),
    # HARD (5)
    Ticket(
        ticket_id=11,
        level="hard",
        message="My account was charged and locked after I tried reset, and the UI says server error.",
        user_history=["password reset loop", "multiple retries"],
        current_status="open",
        urgency_hint="high",
        expected_category="technical",
        expected_priority="high",
        expected_keywords=["account", "charged", "locked", "server error"],
    ),
    Ticket(
        ticket_id=12,
        level="hard",
        message="There is weird language in billing page. I can't verify my identity and ticket replicate multiple issues.",
        user_history=["identity checks failed", "sometimes works"],
        current_status="open",
        urgency_hint="high",
        expected_category="account",
        expected_priority="high",
        expected_keywords=["billing", "identity", "verify", "issue"],
    ),
    Ticket(
        ticket_id=13,
        level="hard",
        message="Site says subscription active yet I'm still billed for old plan; phone number is unverified too.",
        user_history=["plan mismatch", "support follow-up"],
        current_status="open",
        urgency_hint="high",
        expected_category="billing",
        expected_priority="high",
        expected_keywords=["subscription", "billed", "old plan", "unverified"],
    ),
    Ticket(
        ticket_id=14,
        level="hard",
        message="After update, notifications are broken and an error 503 popped while changing account email.",
        user_history=["several errors", "multistep failure"],
        current_status="open",
        urgency_hint="medium",
        expected_category="technical",
        expected_priority="medium",
        expected_keywords=["503", "notifications", "email", "error"],
    ),
    Ticket(
        ticket_id=15,
        level="hard",
        message="I requested cancellation but still charged and there are conflicting instructions from automated system.",
        user_history=["cancellation pending", "automated commands"],
        current_status="open",
        urgency_hint="high",
        expected_category="billing",
        expected_priority="high",
        expected_keywords=["cancel", "charged", "conflicting", "confirmation"],
    ),
]


def get_tickets(level: str) -> List[Ticket]:
    return [ticket for ticket in TICKETS if ticket.level == level]


def get_ticket_by_id(ticket_id: int) -> Optional[Ticket]:
    for ticket in TICKETS:
        if ticket.ticket_id == ticket_id:
            return ticket
    return None
