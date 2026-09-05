"""
Gmail Smart Sorter V7.1 — Semantic Classifier with Persistent Cache.

Key V7.1 features:
- BAAI/bge-m3 embedding model
- Qdrant persistent embedding cache (avoid re-encoding duplicate texts)
- Deterministic SHA-256 cache keys
- Subject-first weighted signals (subject 0.65, full-text 0.35)
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
from embedding_store import EmbeddingStore, generate_cache_key
from models import CategoryScore, ClassificationResult, EmailInput

logger = logging.getLogger(__name__)

# --- Family Hierarchy (unchanged) ---
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

CATEGORY_TO_FAMILY: Dict[str, str] = {}
for family, cats in FAMILY_MAP.items():
    for cat in cats:
        CATEGORY_TO_FAMILY[cat] = family

ALL_CATEGORIES = list(CATEGORY_TO_FAMILY.keys())

# --- Deterministic Override Patterns ---
OVERRIDE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"someone\s+signed\s+in(?:to)?\s+your\s+account", re.I), "Security"),
    (re.compile(r"new\s+sign[\-\s]?in\s+(?:on|from|to|detected)", re.I), "Security"),
    (re.compile(r"password\s+(?:was\s+)?(?:changed|reset)", re.I), "Security"),
    (re.compile(r"suspicious\s+(?:sign[\-\s]?in|login|activity)", re.I), "Security"),
    (re.compile(r"unauthorized\s+access", re.I), "Security"),
    (re.compile(r"github\s+actions?", re.I), "Development/GitHub"),
    (re.compile(r"pull\s+request\s+(?:#\d+|opened|merged|closed|review)", re.I), "Development/GitHub"),
    (re.compile(r"\[[\w\-]+/[\w\-]+\]\s+(?:Run\s+failed|Build|Deploy)", re.I), "Development/GitHub"),
    (re.compile(r"interview\s+(?:scheduled|confirmed|invitation)", re.I), "Jobs/Interviews"),
    (re.compile(r"your\s+(?:technical\s+)?interview\s+is\s+scheduled", re.I), "Jobs/Interviews"),
    (re.compile(r"credit\s+card\s+statement", re.I), "Finance/Credit Cards"),
    (re.compile(r"UPI\s+(?:payment|transaction|transfer|collect)", re.I), "Finance/UPI"),
]

# --- Complaint Signals ---
COMPLAINT_SIGNALS = re.compile(
    r"complaint|installation\s+(?:charges|practices)|misleading\s+warranty|"
    r"technician|service\s+issue|defective|faulty|poor\s+service|"
    r"consumer\s+forum|grievance|dissatisfied",
    re.I
)


class EmailClassifier:
    def __init__(self) -> None:
        self.model = None
        self.model_name: str = config.MODEL_NAME
        self.category_names: List[str] = []
        self._is_loaded: bool = False
        
        # BGE-M3 outputs 1024 dimensions
        self.dimension = 1024 
        self.store = EmbeddingStore(dimension=self.dimension)

        self.prototype_embeddings: Dict[str, np.ndarray] = {}
        self.total_prototype_count: int = 0

        self._lock = threading.RLock()
        self.last_used_time: float = 0.0
        self._stop_watcher = threading.Event()
        self._watcher_thread: Optional[threading.Thread] = None

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._is_loaded

    def start_watcher(self) -> None:
        if self._watcher_thread is None or not self._watcher_thread.is_alive():
            self._stop_watcher.clear()
            self._watcher_thread = threading.Thread(
                target=self._idle_watcher_loop, daemon=True
            )
            self._watcher_thread.start()
            logger.info("Memory idle-watcher started (timeout: %ds).", config.IDLE_TIMEOUT_SECONDS)

    def stop_watcher(self) -> None:
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
        self.model = None
        self._is_loaded = False
        gc.collect()
        logger.info("Model unloaded. RAM freed.")

    def load(self) -> None:
        with self._lock:
            if self._is_loaded:
                self.last_used_time = time.time()
                return

            logger.info("Loading sentence transformer model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer
            # Add some optimizations for CPU if applicable
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")

            self._load_and_cache_prototypes()
            self._is_loaded = True
            self.last_used_time = time.time()
            logger.info(
                "V7.1 classifier ready. %d categories, %d prototypes loaded.",
                len(self.category_names), self.total_prototype_count,
            )

    def _load_and_cache_prototypes(self) -> None:
        """Load prototypes, compute missing embeddings, and cache in Qdrant."""
        proto_path = Path(config.PROTOTYPES_FILE)
        if not proto_path.is_absolute():
            proto_path = Path(__file__).parent / proto_path

        with open(proto_path, "r", encoding="utf-8") as f:
            prototype_data: Dict[str, List[str]] = json.load(f)

        self.category_names = list(prototype_data.keys())
        
        all_texts: List[str] = []
        text_to_category: List[str] = []
        
        for category, descriptions in prototype_data.items():
            for desc in descriptions:
                all_texts.append(desc)
                text_to_category.append(category)

        self.total_prototype_count = len(all_texts)
        
        # Batch get from cache
        keys = [generate_cache_key(text) for text in all_texts]
        cached_vectors = self.store.get_batch(keys)
        
        missing_indices = []
        missing_texts = []
        missing_keys = []
        missing_metas = []
        
        for i, text in enumerate(all_texts):
            key = keys[i]
            if key not in cached_vectors:
                missing_indices.append(i)
                missing_texts.append(text)
                missing_keys.append(key)
                missing_metas.append({"type": "prototype", "category": text_to_category[i]})

        if missing_texts:
            logger.info("Encoding %d missing prototypes...", len(missing_texts))
            new_embeddings = self.model.encode(
                missing_texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Store in cache
            self.store.put_batch(missing_keys, new_embeddings.tolist(), missing_metas)
            
            # Update cache dict
            for i, key in enumerate(missing_keys):
                cached_vectors[key] = new_embeddings[i].tolist()

        # Group by category
        category_embeds: Dict[str, List[np.ndarray]] = {cat: [] for cat in self.category_names}
        for i, cat in enumerate(text_to_category):
            key = keys[i]
            category_embeds[cat].append(np.array(cached_vectors[key], dtype=np.float32))

        self.prototype_embeddings = {
            cat: np.array(embeds, dtype=np.float32)
            for cat, embeds in category_embeds.items()
            if embeds
        }
        logger.info("Prototype embeddings initialized.")

    def _get_or_encode_texts(self, texts: List[str], metadatas: List[Dict]) -> List[np.ndarray]:
        """Fetch embeddings from Qdrant or encode and store if missing."""
        keys = [generate_cache_key(text) for text in texts]
        cached_vectors = self.store.get_batch(keys)
        
        missing_indices = []
        missing_texts = []
        missing_keys = []
        missing_metas = []
        
        for i, text in enumerate(texts):
            key = keys[i]
            if key not in cached_vectors:
                missing_indices.append(i)
                missing_texts.append(text)
                missing_keys.append(key)
                missing_metas.append(metadatas[i])
                
        if missing_texts:
            new_embeddings = self.model.encode(
                missing_texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.store.put_batch(missing_keys, new_embeddings.tolist(), missing_metas)
            for i, key in enumerate(missing_keys):
                cached_vectors[key] = new_embeddings[i].tolist()
                
        # Reconstruct exactly in original order
        result = []
        for key in keys:
            result.append(np.array(cached_vectors[key], dtype=np.float32))
        return result

    def classify_batch(self, emails: List[EmailInput]) -> List[ClassificationResult]:
        with self._lock:
            if not self._is_loaded:
                self.load()
            self.last_used_time = time.time()

            subject_texts = []
            body_texts = []
            subject_metas = []
            body_metas = []

            for email in emails:
                clean_body = self._clean_body(email.body)
                subj = f"SUBJECT: {email.subject}" if email.subject else "SUBJECT: (empty)"
                body = f"BODY: {clean_body}" if clean_body else "BODY: (empty)"
                
                subject_texts.append(subj)
                body_texts.append(body)
                
                subject_metas.append({"type": "subject", "email_id": email.id})
                body_metas.append({"type": "body", "email_id": email.id})

            subject_embeds = self._get_or_encode_texts(subject_texts, subject_metas)
            body_embeds = self._get_or_encode_texts(body_texts, body_metas)

            results: List[ClassificationResult] = []
            for i, email in enumerate(emails):
                result = self._classify_single(
                    email_id=email.id,
                    subject=email.subject,
                    body=email.body,
                    subject_embed=subject_embeds[i],
                    body_embed=body_embeds[i],
                )
                results.append(result)

            return results

    def _classify_single(
        self,
        email_id: str,
        subject: str,
        body: str,
        subject_embed: np.ndarray,
        body_embed: np.ndarray,
    ) -> ClassificationResult:
        combined_text = f"{subject} {body}"

        # 1. Deterministic overrides
        override = self._check_overrides(subject, body)
        if override:
            return self._build_override_result(email_id, override, subject_embed, body_embed)

        # 2. Complaint safety
        is_complaint = bool(COMPLAINT_SIGNALS.search(combined_text))

        # 3. Scoring
        category_scores: Dict[str, float] = {}
        category_subj_scores: Dict[str, float] = {}
        category_body_scores: Dict[str, float] = {}
        category_support: Dict[str, float] = {}
        
        for cat_name, proto_embeds in self.prototype_embeddings.items():
            s_score, s_support = self._score_signal(proto_embeds, subject_embed)
            b_score, b_support = self._score_signal(proto_embeds, body_embed)
            
            final_score = config.WEIGHT_SUBJECT * s_score + config.WEIGHT_BODY * b_score
            
            if is_complaint and CATEGORY_TO_FAMILY.get(cat_name) == "Finance":
                final_score *= 0.3
                s_score *= 0.3
                b_score *= 0.3
                
            category_scores[cat_name] = final_score
            category_subj_scores[cat_name] = s_score
            category_body_scores[cat_name] = b_score
            category_support[cat_name] = (s_support + b_support) / 2.0

        # 4. Hierarchical resolution
        family_scores = self._compute_family_scores(category_scores)
        best_family = max(family_scores, key=family_scores.get)
        best_family_score = family_scores[best_family]

        sorted_families = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
        second_family_score = sorted_families[1][1] if len(sorted_families) > 1 else 0.0
        family_margin = best_family_score - second_family_score

        family_cats = FAMILY_MAP.get(best_family, [])
        subcategory_scores = {cat: category_scores.get(cat, 0.0) for cat in family_cats}
        best_category = max(subcategory_scores, key=subcategory_scores.get)
        best_score = subcategory_scores[best_category]

        sorted_sub = sorted(subcategory_scores.values(), reverse=True)
        sub_margin = sorted_sub[0] - sorted_sub[1] if len(sorted_sub) > 1 else sorted_sub[0]

        # 5. Conflicts
        best_category, reason = self._resolve_conflicts(
            best_category, best_family, category_scores, family_scores,
            subject, body, family_margin
        )
        
        best_family = CATEGORY_TO_FAMILY.get(best_category, best_family)
        best_score = category_scores.get(best_category, best_score)
        best_subj_score = category_subj_scores.get(best_category, 0.0)
        best_body_score = category_body_scores.get(best_category, 0.0)
        best_support = category_support.get(best_category, 0.0)

        # 6. Top-3
        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [
            CategoryScore(category=cat, score=round(score, 4))
            for cat, score in sorted_cats[:config.TOP_K]
        ]

        # 7. Overall margin
        margin = sorted_cats[0][1] - sorted_cats[1][1] if len(sorted_cats) > 1 else sorted_cats[0][1]

        # 8. Confidence
        confidence = self._calibrate_confidence(
            best_score, margin, best_family_score, family_margin, sub_margin, best_support
        )

        # 9. Decision
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
            subject_score=round(best_subj_score, 4),
            body_score=round(best_body_score, 4),
            prototype_support=round(best_support, 4),
            family=best_family,
            family_confidence=round(best_family_score, 4),
            family_margin=round(family_margin, 4),
            top3=top3,
            reason=reason,
        )

    def _score_signal(self, proto_embeds: np.ndarray, embed: np.ndarray) -> Tuple[float, float]:
        """Returns (weighted_score, prototype_support)"""
        sims = np.dot(proto_embeds, embed)
        sorted_sims = np.sort(sims)[::-1]
        
        top1 = float(sorted_sims[0])
        k = min(config.PROTO_TOP_K, len(sorted_sims))
        topk_mean = float(np.mean(sorted_sims[:k]))
        
        # prototype_support represents how closely the top-k agree with the top-1
        support = topk_mean / max(top1, 0.01) if top1 > 0 else 0.0
        
        score = config.PROTO_TOP1_WEIGHT * top1 + config.PROTO_TOPK_WEIGHT * topk_mean
        return score, support

    def _compute_family_scores(self, category_scores: Dict[str, float]) -> Dict[str, float]:
        family_scores: Dict[str, float] = {}
        for family, cats in FAMILY_MAP.items():
            scores = [category_scores.get(cat, 0.0) for cat in cats]
            family_scores[family] = max(scores) if scores else 0.0
        return family_scores

    def _resolve_conflicts(
        self,
        best_category: str,
        best_family: str,
        category_scores: Dict[str, float],
        family_scores: Dict[str, float],
        subject: str,
        body: str,
        family_margin: float,
    ) -> Tuple[str, str]:
        combined = f"{subject} {body}".lower()
        reason = ""

        # Jobs vs Development
        if best_family in ("Jobs", "Development"):
            jobs_score = family_scores.get("Jobs", 0)
            dev_score = family_scores.get("Development", 0)
            if abs(jobs_score - dev_score) < 0.08:
                job_signals = re.search(
                    r"hiring|opening|position|vacancy|apply|job\s+alert|"
                    r"engineer(?:ing)?\s+(?:at|@)|developer\s+(?:at|@|-\s*remote)|"
                    r"intern(?:ship)?|roles?\s+(?:at|–|apply)|requirement\s+for",
                    combined, re.I
                )
                dev_signals = re.search(
                    r"released?|version\s+\d|changelog|release\s+notes|"
                    r"api\s+pricing|sdk\s+update|framework\s+update|documentation\s+update",
                    combined, re.I
                )
                if job_signals and not dev_signals:
                    best_category = max([c for c in category_scores if c.startswith("Jobs/")], key=lambda c: category_scores[c])
                    reason = "Resolved Jobs vs Development conflict: job-specific language detected"
                elif dev_signals and not job_signals:
                    best_category = max([c for c in category_scores if c.startswith("Development/")], key=lambda c: category_scores[c])
                    reason = "Resolved Jobs vs Development conflict: programming/release language detected"

        # Social vs Jobs/Recruiters
        if best_category in ("Jobs/Recruiters", "Social"):
            vague = re.search(r"^i\s+want\s+to\s+connect$", subject.strip(), re.I)
            if vague:
                best_category = "Review"
                reason = "Ambiguous connection request redirected to Review"

        # Finance vs Complaint
        if best_family == "Finance" and COMPLAINT_SIGNALS.search(combined):
            best_category = "Review"
            reason = "Complaint language detected; Finance classification suppressed"

        # Domain expiry
        domain_expiry = re.search(r"domain.*expir|expir.*domain|keep\s+your\s+domain|domain.*expired", combined, re.I)
        if domain_expiry and best_family == "Development":
            best_category = "Review"
            reason = "Domain expiry notification redirected to Review"

        # Google Wallet welcome
        wallet_welcome = re.search(r"(?:google\s+wallet|welcome\s+to\s+google\s+wallet)", combined, re.I)
        if wallet_welcome and best_family in ("Finance", "Travel"):
            best_category = "Review"
            reason = "Google Wallet welcome message redirected to Review"

        # Learning vs Jobs
        if best_family == "Jobs":
            learning_course = re.search(
                r"course|enroll|degree|certification|bootcamp|training\s+program|"
                r"learn\s+(?:ai|ml|python|programming)|skill\s+development",
                combined, re.I
            )
            if learning_course and not re.search(r"hiring|opening|position|vacancy|apply\s+(?:now|today)", combined, re.I):
                best_category = max([c for c in category_scores if c.startswith("Learning/")], key=lambda c: category_scores[c])
                reason = "Resolved Jobs vs Learning conflict: course/education language detected"
                best_family = "Learning"

        # Promotions masquerading as Finance/UPI
        if best_category == "Finance/UPI":
            marketing = re.search(r"meet\s+your\s+upi's\s+best\s+friend|earn\s+rewards\s+on\s+upi|cashback\s+on\s+upi", combined, re.I)
            if marketing:
                best_category = "Promotions"
                reason = "UPI marketing email redirected to Promotions"

        # Development masquerading as Security
        if best_category == "Security":
            dev_verify = re.search(r"developer\s+verification|android\s+developer|app\s+verification", combined, re.I)
            if dev_verify:
                best_category = "Development/Programming"
                reason = "Developer verification redirected to Development"

        return best_category, reason

    def _check_overrides(self, subject: str, body: str) -> Optional[str]:
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
        body_embed: np.ndarray,
    ) -> ClassificationResult:
        family = CATEGORY_TO_FAMILY.get(category, "Review")
        return ClassificationResult(
            id=email_id,
            category=category,
            decision="AUTO_SORT",
            confidence=0.98,
            similarity=0.95,
            margin=0.3,
            subject_score=0.95,
            body_score=0.95,
            prototype_support=1.0,
            family=family,
            family_confidence=0.98,
            family_margin=0.3,
            top3=[CategoryScore(category=category, score=0.95)],
            reason=f"Deterministic override: matched high-precision pattern for {category}",
        )

    def _calibrate_confidence(
        self,
        best_score: float,
        margin: float,
        family_score: float,
        family_margin: float,
        sub_margin: float,
        prototype_support: float
    ) -> float:
        # Normalize each factor to [0, 1]
        score_factor = min(max(best_score, 0.0), 1.0)
        margin_factor = min(margin / 0.15, 1.0)
        family_factor = min(max(family_score, 0.0), 1.0)
        fam_margin_factor = min(family_margin / 0.15, 1.0)
        support_factor = min(max(prototype_support, 0.0), 1.0)

        confidence = (
            0.25 * score_factor
            + 0.25 * margin_factor
            + 0.20 * family_factor
            + 0.15 * fam_margin_factor
            + 0.15 * support_factor
        )

        return max(0.0, min(1.0, confidence))

    def _decide(self, confidence: float, margin: float) -> str:
        if confidence >= config.AUTO_SORT_CONFIDENCE and margin >= config.AUTO_SORT_MARGIN:
            return "AUTO_SORT"
        elif confidence >= config.REVIEW_CONFIDENCE:
            return "REVIEW"
        else:
            return "UNMATCHED"

    def _generate_reason(
        self, category: str, confidence: float, margin: float, family_margin: float
    ) -> str:
        family = CATEGORY_TO_FAMILY.get(category, "Unknown")
        if confidence >= 0.85 and margin >= 0.10:
            return f"Strong semantic match to {category} prototypes (family: {family})"
        elif confidence >= 0.70:
            return f"Good match to {category} prototypes with moderate confidence"
        elif confidence >= 0.55:
            return f"Weak match to {category}; consider manual review"
        else:
            return f"Low confidence match to {category}; classification uncertain"

    def _clean_body(self, body: str) -> str:
        if not body:
            return ""
        text = body[:config.MAX_BODY_LENGTH]
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        text = re.sub(r"utm_\w+=\S+", " ", text)
        text = re.sub(r"(?i)(?:unsubscribe|opt[\s-]?out|manage\s+preferences).*$", "", text)
        text = re.sub(r"(?m)^>.*$", "", text)
        text = re.sub(r"(?i)on\s+\w+\s+\d+.*wrote:", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @property
    def num_categories(self) -> int:
        return len(self.category_names)

    @property
    def num_prototypes(self) -> int:
        return self.total_prototype_count

classifier = EmailClassifier()
