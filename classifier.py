"""
Gmail Smart Sorter V7 — Multi-Prototype Hierarchical Classifier.

Key V7 improvements over V6:
- Multi-prototype scoring (top-1 + top-K mean, not centroids)
- Subject-first weighted signals (subject 0.5, full-text 0.4, sender 0.1)
- Hierarchical family → subcategory resolution
- Conflict detection for confusing boundaries
- Deterministic overrides for high-precision cases
- Calibrated confidence (not raw cosine similarity)
"""

import gc
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from models import CategoryScore, ClassificationResult, EmailInput

logger = logging.getLogger(__name__)

# --- Family Hierarchy ---
FAMILY_MAP: Dict[str, List[str]] = {
    "Security": ["Security"],
    "Finance": ["Finance/UPI", "Finance/Credit Cards", "Finance/Investments", "Finance/Banks", "Finance/Bills"],
    "Jobs": ["Jobs/Job Alerts", "Jobs/Interviews", "Jobs/Recruiters"],
    "Development": ["Development/GitHub", "Development/Server Alerts", "Development/Programming"],
    "Learning": ["Learning/Courses", "Learning/Events", "Learning/AI-Tech"],
    "Shopping": ["Shopping/Amazon", "Shopping/Flipkart", "Shopping/Other"],
    "Travel": ["Travel"],
    "Social": ["Social"],
    "Newsletters": ["Newsletters"],
    "Promotions": ["Promotions"],
    "Work": ["Work", "Work/Reports", "Work/HR"],
    "Review": ["Review"],
}

# Build reverse map: category → family
CATEGORY_TO_FAMILY: Dict[str, str] = {}
for family, cats in FAMILY_MAP.items():
    for cat in cats:
        CATEGORY_TO_FAMILY[cat] = family

ALL_CATEGORIES = list(CATEGORY_TO_FAMILY.keys())

# --- Deterministic Override Patterns ---
# Each pattern: (compiled_regex, required_category)
# These fire ONLY when the match is unambiguous.
OVERRIDE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Security - require actual security event semantics
    (re.compile(r"someone\s+signed\s+in(?:to)?\s+your\s+account", re.I), "Security"),
    (re.compile(r"new\s+sign[\-\s]?in\s+(?:on|from|to|detected)", re.I), "Security"),
    (re.compile(r"password\s+(?:was\s+)?(?:changed|reset)", re.I), "Security"),
    (re.compile(r"suspicious\s+(?:sign[\-\s]?in|login|activity)", re.I), "Security"),
    (re.compile(r"unauthorized\s+access", re.I), "Security"),

    # GitHub - require GitHub-specific terms
    (re.compile(r"github\s+actions?", re.I), "Development/GitHub"),
    (re.compile(r"pull\s+request\s+(?:#\d+|opened|merged|closed|review)", re.I), "Development/GitHub"),
    (re.compile(r"\[[\w\-]+/[\w\-]+\]\s+(?:Run\s+failed|Build|Deploy)", re.I), "Development/GitHub"),

    # Interviews - require interview-specific language
    (re.compile(r"interview\s+(?:scheduled|confirmed|invitation)", re.I), "Jobs/Interviews"),
    (re.compile(r"your\s+(?:technical\s+)?interview\s+is\s+scheduled", re.I), "Jobs/Interviews"),

    # Credit cards - require explicit credit card statement
    (re.compile(r"credit\s+card\s+statement", re.I), "Finance/Credit Cards"),

    # UPI - require explicit UPI mention
    (re.compile(r"UPI\s+(?:payment|transaction|transfer|collect)", re.I), "Finance/UPI"),
]

# --- Complaint / Non-Finance Signals ---
COMPLAINT_SIGNALS = re.compile(
    r"complaint|installation\s+(?:charges|practices)|misleading\s+warranty|"
    r"technician|service\s+issue|defective|faulty|poor\s+service|"
    r"consumer\s+forum|grievance|dissatisfied",
    re.I
)


class EmailClassifier:
    """
    V7 Multi-Prototype Hierarchical Email Classifier.

    Loads prototypes at startup, encodes them once, then classifies
    incoming emails using weighted multi-signal similarity with
    hierarchical family→subcategory resolution.
    """

    def __init__(self) -> None:
        self.model = None
        self.model_name: str = config.MODEL_NAME
        self.category_names: List[str] = []
        self._is_loaded: bool = False

        # Per-category prototype embeddings: {category: np.ndarray of shape (N, dim)}
        self.prototype_embeddings: Dict[str, np.ndarray] = {}
        self.total_prototype_count: int = 0

        # Thread safety and dynamic memory management
        self._lock = threading.RLock()
        self.last_used_time: float = 0.0
        self._stop_watcher = threading.Event()
        self._watcher_thread: Optional[threading.Thread] = None

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._is_loaded

    def start_watcher(self) -> None:
        """Starts the background thread that unloads the model when idle."""
        if self._watcher_thread is None or not self._watcher_thread.is_alive():
            self._stop_watcher.clear()
            self._watcher_thread = threading.Thread(
                target=self._idle_watcher_loop, daemon=True
            )
            self._watcher_thread.start()
            logger.info("Memory idle-watcher started (timeout: %ds).", config.IDLE_TIMEOUT_SECONDS)

    def stop_watcher(self) -> None:
        """Stops the background thread."""
        self._stop_watcher.set()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=2.0)
            self._watcher_thread = None

    def _idle_watcher_loop(self) -> None:
        while not self._stop_watcher.is_set():
            time.sleep(5)
            with self._lock:
                if self._is_loaded and (time.time() - self.last_used_time > config.IDLE_TIMEOUT_SECONDS):
                    logger.info("Idle timeout reached. Unloading model to free RAM.")
                    self._unload_unsafe()

    def _unload_unsafe(self) -> None:
        """Unloads the model without acquiring the lock (caller must hold it)."""
        self.model = None
        self.prototype_embeddings = {}
        self._is_loaded = False
        gc.collect()
        logger.info("Model unloaded. RAM freed.")

    def load(self) -> None:
        """Load the model and pre-compute prototype embeddings. Thread-safe."""
        with self._lock:
            if self._is_loaded:
                self.last_used_time = time.time()
                return

            logger.info("Loading sentence transformer model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")

            self._load_prototypes()
            self._is_loaded = True
            self.last_used_time = time.time()
            logger.info(
                "V7 classifier ready. %d categories, %d prototypes loaded.",
                len(self.category_names), self.total_prototype_count,
            )

    def _load_prototypes(self) -> None:
        """Load prototypes.json and pre-compute normalized prototype embeddings."""
        proto_path = Path(config.PROTOTYPES_FILE)
        if not proto_path.is_absolute():
            proto_path = Path(__file__).parent / proto_path

        if not proto_path.exists():
            raise FileNotFoundError(f"Prototypes file not found: {proto_path}")

        with open(proto_path, "r", encoding="utf-8") as f:
            prototype_data: Dict[str, List[str]] = json.load(f)

        self.category_names = list(prototype_data.keys())

        # Validate all categories exist in the hierarchy
        for cat in self.category_names:
            if cat not in CATEGORY_TO_FAMILY:
                logger.warning("Category '%s' not found in family hierarchy.", cat)

        # Collect all texts for batch encoding
        all_texts: List[str] = []
        text_to_category: List[str] = []
        for category, descriptions in prototype_data.items():
            for desc in descriptions:
                all_texts.append(desc)
                text_to_category.append(category)

        logger.info("Encoding %d prototypes across %d categories...", len(all_texts), len(self.category_names))

        # Batch-encode all prototypes at once (normalized)
        all_embeddings = self.model.encode(
            all_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Group embeddings by category
        category_embeds: Dict[str, List[np.ndarray]] = {cat: [] for cat in self.category_names}
        for i, cat in enumerate(text_to_category):
            category_embeds[cat].append(all_embeddings[i])

        # Stack into per-category arrays
        self.prototype_embeddings = {
            cat: np.array(embeds, dtype=np.float32)
            for cat, embeds in category_embeds.items()
            if embeds
        }

        self.total_prototype_count = len(all_texts)
        logger.info("Prototype embeddings computed and cached.")

    # ─────────────── Classification Pipeline ───────────────

    def classify_batch(self, emails: List[EmailInput]) -> List[ClassificationResult]:
        """Classify a batch of emails. Thread-safe."""
        with self._lock:
            if not self._is_loaded:
                self.load()
            self.last_used_time = time.time()

            # Build three text signals per email
            subject_texts = []
            full_texts = []
            sender_texts = []

            for email in emails:
                clean_body = self._clean_body(email.body)
                subject_texts.append(f"SUBJECT: {email.subject}" if email.subject else "SUBJECT: (empty)")
                full_texts.append(self._build_full_text(email.subject, email.sender, clean_body))
                sender_texts.append(f"SENDER: {email.sender}" if email.sender else "SENDER: (unknown)")

            # Batch encode all three signals
            subject_embeds = self.model.encode(
                subject_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )
            full_embeds = self.model.encode(
                full_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )
            sender_embeds = self.model.encode(
                sender_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )

            # Classify each email
            results: List[ClassificationResult] = []
            for i, email in enumerate(emails):
                result = self._classify_single(
                    email_id=email.id,
                    subject=email.subject,
                    sender=email.sender,
                    body=email.body,
                    subject_embed=subject_embeds[i],
                    full_embed=full_embeds[i],
                    sender_embed=sender_embeds[i],
                )
                results.append(result)

            return results

    def classify_text(self, text: str) -> ClassificationResult:
        """Classify a single text string (for /evaluate). Thread-safe."""
        with self._lock:
            if not self._is_loaded:
                self.load()
            self.last_used_time = time.time()

            # Use text as both subject and full text
            subject_embed = self.model.encode(
                [f"SUBJECT: {text}"], normalize_embeddings=True, show_progress_bar=False
            )[0]
            full_embed = self.model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0]
            sender_embed = self.model.encode(
                ["SENDER: (unknown)"], normalize_embeddings=True, show_progress_bar=False
            )[0]

            return self._classify_single(
                email_id="eval",
                subject=text,
                sender="",
                body="",
                subject_embed=subject_embed,
                full_embed=full_embed,
                sender_embed=sender_embed,
            )

    def _classify_single(
        self,
        email_id: str,
        subject: str,
        sender: str,
        body: str,
        subject_embed: np.ndarray,
        full_embed: np.ndarray,
        sender_embed: np.ndarray,
    ) -> ClassificationResult:
        """Core classification pipeline for a single email."""

        combined_text = f"{subject} {sender} {body}"

        # Step 1: Check deterministic overrides
        override = self._check_overrides(subject, body)
        if override:
            return self._build_override_result(email_id, override, subject_embed, full_embed, sender_embed)

        # Step 2: Check complaint safety (must come before scoring)
        is_complaint = bool(COMPLAINT_SIGNALS.search(combined_text))

        # Step 3: Multi-prototype scoring for each category
        category_scores: Dict[str, float] = {}
        for cat_name, proto_embeds in self.prototype_embeddings.items():
            score = self._compute_category_score(
                proto_embeds, subject_embed, full_embed, sender_embed
            )

            # Complaint penalty: dramatically reduce Finance scores
            if is_complaint and CATEGORY_TO_FAMILY.get(cat_name) == "Finance":
                score *= 0.3

            category_scores[cat_name] = score

        # Step 4: Hierarchical resolution
        family_scores = self._compute_family_scores(category_scores)
        best_family = max(family_scores, key=family_scores.get)
        best_family_score = family_scores[best_family]

        # Sort families for margin
        sorted_families = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
        second_family_score = sorted_families[1][1] if len(sorted_families) > 1 else 0.0
        family_margin = best_family_score - second_family_score

        # Get best subcategory within the winning family
        family_cats = FAMILY_MAP.get(best_family, [])
        subcategory_scores = {cat: category_scores.get(cat, 0.0) for cat in family_cats}
        best_category = max(subcategory_scores, key=subcategory_scores.get)
        best_score = subcategory_scores[best_category]

        # Subcategory margin within family
        sorted_sub = sorted(subcategory_scores.values(), reverse=True)
        sub_margin = sorted_sub[0] - sorted_sub[1] if len(sorted_sub) > 1 else sorted_sub[0]

        # Step 5: Conflict detection
        best_category, reason = self._resolve_conflicts(
            best_category, best_family, category_scores, family_scores,
            subject, sender, body, family_margin
        )
        # Recompute family after potential conflict resolution
        best_family = CATEGORY_TO_FAMILY.get(best_category, best_family)
        best_score = category_scores.get(best_category, best_score)

        # Step 6: Top-3 categories (across all categories)
        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [
            CategoryScore(category=cat, score=round(score, 4))
            for cat, score in sorted_cats[:config.TOP_K]
        ]

        # Step 7: Overall margin (across all categories)
        margin = sorted_cats[0][1] - sorted_cats[1][1] if len(sorted_cats) > 1 else sorted_cats[0][1]

        # Step 8: Confidence calibration
        confidence = self._calibrate_confidence(
            best_score, margin, best_family_score, family_margin, sub_margin
        )

        # Step 9: Decision
        decision = self._decide(confidence, margin)

        if not reason:
            reason = self._generate_reason(best_category, confidence, margin, family_margin)

        return ClassificationResult(
            id=email_id,
            category=best_category,
            decision=decision,
            confidence=round(confidence, 4),
            similarity=round(best_score, 4),
            margin=round(margin, 4),
            family=best_family,
            family_confidence=round(best_family_score, 4),
            family_margin=round(family_margin, 4),
            top3=top3,
            reason=reason,
        )

    # ─────────────── Scoring ───────────────

    def _compute_category_score(
        self,
        proto_embeds: np.ndarray,
        subject_embed: np.ndarray,
        full_embed: np.ndarray,
        sender_embed: np.ndarray,
    ) -> float:
        """
        Multi-prototype category score using weighted signals.

        For each signal, compute similarity against every prototype,
        then combine top-1 and top-K mean. Finally weight the signals.
        """
        # Subject signal
        subject_sims = np.dot(proto_embeds, subject_embed)
        subject_score = self._top_k_score(subject_sims)

        # Full-text signal
        full_sims = np.dot(proto_embeds, full_embed)
        full_score = self._top_k_score(full_sims)

        # Sender signal
        sender_sims = np.dot(proto_embeds, sender_embed)
        sender_score = self._top_k_score(sender_sims)

        # Weighted combination
        return (
            config.WEIGHT_SUBJECT * subject_score
            + config.WEIGHT_FULLTEXT * full_score
            + config.WEIGHT_SENDER * sender_score
        )

    def _top_k_score(self, similarities: np.ndarray) -> float:
        """
        Compute score from prototype similarities:
        0.70 * max(sims) + 0.30 * mean(top_3(sims))
        """
        sorted_sims = np.sort(similarities)[::-1]
        top1 = float(sorted_sims[0])
        k = min(config.PROTO_TOP_K, len(sorted_sims))
        topk_mean = float(np.mean(sorted_sims[:k]))
        return config.PROTO_TOP1_WEIGHT * top1 + config.PROTO_TOPK_WEIGHT * topk_mean

    def _compute_family_scores(self, category_scores: Dict[str, float]) -> Dict[str, float]:
        """Aggregate subcategory scores into family scores (max of subcategories)."""
        family_scores: Dict[str, float] = {}
        for family, cats in FAMILY_MAP.items():
            scores = [category_scores.get(cat, 0.0) for cat in cats]
            family_scores[family] = max(scores) if scores else 0.0
        return family_scores

    # ─────────────── Conflict Detection ───────────────

    def _resolve_conflicts(
        self,
        best_category: str,
        best_family: str,
        category_scores: Dict[str, float],
        family_scores: Dict[str, float],
        subject: str,
        sender: str,
        body: str,
        family_margin: float,
    ) -> Tuple[str, str]:
        """
        Detect and resolve known classification conflicts.
        Returns (resolved_category, reason_string).
        """
        combined = f"{subject} {body}".lower()
        reason = ""

        # Jobs vs Development: "Python engineer opening" → Jobs, "Python 3.14 Released" → Dev
        if best_family in ("Jobs", "Development"):
            jobs_score = family_scores.get("Jobs", 0)
            dev_score = family_scores.get("Development", 0)
            if abs(jobs_score - dev_score) < 0.05:
                # Check for job-specific signals
                job_signals = re.search(
                    r"hiring|opening|position|vacancy|apply|job\s+alert|"
                    r"engineer(?:ing)?\s+(?:at|@)|developer\s+(?:at|@|-\s*remote)|"
                    r"intern(?:ship)?|roles?\s+(?:at|–|apply)|requirement\s+for",
                    combined, re.I
                )
                dev_signals = re.search(
                    r"released?|version\s+\d|changelog|release\s+notes|"
                    r"api\s+pricing|sdk\s+update|framework\s+update",
                    combined, re.I
                )
                if job_signals and not dev_signals:
                    best_category = max(
                        [c for c in category_scores if c.startswith("Jobs/")],
                        key=lambda c: category_scores[c],
                    )
                    reason = "Resolved Jobs vs Development conflict: job-specific language detected"
                elif dev_signals and not job_signals:
                    best_category = max(
                        [c for c in category_scores if c.startswith("Development/")],
                        key=lambda c: category_scores[c],
                    )
                    reason = "Resolved Jobs vs Development conflict: programming/release language detected"

        # Social vs Jobs/Recruiters: "I want to connect" → Review
        if best_category == "Jobs/Recruiters" or best_category == "Social":
            vague = re.search(r"^i\s+want\s+to\s+connect$", subject.strip(), re.I)
            if vague:
                best_category = "Review"
                reason = "Ambiguous connection request redirected to Review"

        # Finance vs Complaint: complaint language overrides Finance
        if best_family == "Finance" and COMPLAINT_SIGNALS.search(combined):
            best_category = "Review"
            reason = "Complaint language detected; Finance classification suppressed"

        # Domain expiry → should be Review, not Development
        domain_expiry = re.search(
            r"domain.*expir|expir.*domain|keep\s+your\s+domain|domain.*expired",
            combined, re.I
        )
        if domain_expiry and best_family == "Development":
            best_category = "Review"
            reason = "Domain expiry notification redirected to Review"

        # Google Wallet welcome → Review, not Finance/Travel
        wallet_welcome = re.search(r"(?:google\s+wallet|welcome\s+to\s+google\s+wallet)", combined, re.I)
        if wallet_welcome and best_family in ("Finance", "Travel"):
            best_category = "Review"
            reason = "Google Wallet welcome message redirected to Review"

        # Learning vs Jobs: "job-ready course" → Learning
        if best_family == "Jobs":
            learning_course = re.search(
                r"course|enroll|degree|certification|bootcamp|training\s+program|"
                r"learn\s+(?:ai|ml|python|programming)|skill\s+development",
                combined, re.I
            )
            if learning_course and not re.search(r"hiring|opening|position|vacancy|apply\s+(?:now|today)", combined, re.I):
                best_category = max(
                    [c for c in category_scores if c.startswith("Learning/")],
                    key=lambda c: category_scores[c],
                )
                reason = "Resolved Jobs vs Learning conflict: course/education language detected"
                best_family = "Learning"

        return best_category, reason

    # ─────────────── Overrides ───────────────

    def _check_overrides(self, subject: str, body: str) -> Optional[str]:
        """Check deterministic override patterns. Returns category or None."""
        combined = f"{subject} {body}"
        for pattern, category in OVERRIDE_PATTERNS:
            if pattern.search(combined):
                return category
        return None

    def _build_override_result(
        self,
        email_id: str,
        category: str,
        subject_embed: np.ndarray,
        full_embed: np.ndarray,
        sender_embed: np.ndarray,
    ) -> ClassificationResult:
        """Build a high-confidence result from a deterministic override."""
        family = CATEGORY_TO_FAMILY.get(category, "Review")

        # Still compute scores for top-3 display
        category_scores: Dict[str, float] = {}
        for cat_name, proto_embeds in self.prototype_embeddings.items():
            category_scores[cat_name] = self._compute_category_score(
                proto_embeds, subject_embed, full_embed, sender_embed
            )

        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [
            CategoryScore(category=cat, score=round(score, 4))
            for cat, score in sorted_cats[:config.TOP_K]
        ]

        similarity = category_scores.get(category, 0.95)
        margin = sorted_cats[0][1] - sorted_cats[1][1] if len(sorted_cats) > 1 else 0.3

        return ClassificationResult(
            id=email_id,
            category=category,
            decision="AUTO_SORT",
            confidence=0.98,
            similarity=round(similarity, 4),
            margin=round(margin, 4),
            family=family,
            family_confidence=0.98,
            family_margin=round(margin, 4),
            top3=top3,
            reason=f"Deterministic override: matched high-precision pattern for {category}",
        )

    # ─────────────── Confidence ───────────────

    def _calibrate_confidence(
        self,
        best_score: float,
        margin: float,
        family_score: float,
        family_margin: float,
        sub_margin: float,
    ) -> float:
        """
        Multi-factor confidence calibration.

        Incorporates:
        - Absolute winning score (30%)
        - Category margin (25%)
        - Family confidence (20%)
        - Family margin (15%)
        - Subcategory margin within family (10%)
        """
        # Normalize each factor to [0, 1]
        score_factor = min(max(best_score, 0.0), 1.0)
        margin_factor = min(margin / 0.15, 1.0)
        family_factor = min(max(family_score, 0.0), 1.0)
        fam_margin_factor = min(family_margin / 0.15, 1.0)
        sub_margin_factor = min(sub_margin / 0.10, 1.0)

        confidence = (
            0.30 * score_factor
            + 0.25 * margin_factor
            + 0.20 * family_factor
            + 0.15 * fam_margin_factor
            + 0.10 * sub_margin_factor
        )

        return max(0.0, min(1.0, confidence))

    def _decide(self, confidence: float, margin: float) -> str:
        """Assign decision state based on confidence and margin."""
        if confidence >= config.AUTO_SORT_CONFIDENCE and margin >= config.AUTO_SORT_MARGIN:
            return "AUTO_SORT"
        elif confidence >= config.REVIEW_CONFIDENCE:
            return "REVIEW"
        else:
            return "UNMATCHED"

    def _generate_reason(
        self, category: str, confidence: float, margin: float, family_margin: float
    ) -> str:
        """Generate a human-readable reason for the classification."""
        family = CATEGORY_TO_FAMILY.get(category, "Unknown")
        if confidence >= 0.85 and margin >= 0.10:
            return f"Strong semantic match to {category} prototypes (family: {family})"
        elif confidence >= 0.70:
            return f"Good match to {category} prototypes with moderate confidence"
        elif confidence >= 0.55:
            return f"Weak match to {category}; consider manual review"
        else:
            return f"Low confidence match to {category}; classification uncertain"

    # ─────────────── Text Processing ───────────────

    def _build_full_text(self, subject: str, sender: str, body: str) -> str:
        """Build the full text representation for an email."""
        parts = []
        if subject:
            parts.append(f"SUBJECT: {subject}")
        if sender:
            parts.append(f"SENDER: {sender}")
        if body:
            parts.append(f"BODY: {body}")
        return "\n".join(parts) if parts else "Empty email"

    def _clean_body(self, body: str) -> str:
        """Strip noise from email body for better embedding quality."""
        if not body:
            return ""

        text = body[:config.MAX_BODY_LENGTH]

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", " ", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        # Remove tracking parameters
        text = re.sub(r"utm_\w+=\S+", " ", text)
        # Remove unsubscribe blocks
        text = re.sub(r"(?i)(?:unsubscribe|opt[\s-]?out|manage\s+preferences).*$", "", text)
        # Remove quoted replies
        text = re.sub(r"(?m)^>.*$", "", text)
        text = re.sub(r"(?i)on\s+\w+\s+\d+.*wrote:", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ─────────────── Properties ───────────────

    @property
    def num_categories(self) -> int:
        return len(self.category_names)

    @property
    def num_prototypes(self) -> int:
        return self.total_prototype_count


# Module-level singleton
classifier = EmailClassifier()
