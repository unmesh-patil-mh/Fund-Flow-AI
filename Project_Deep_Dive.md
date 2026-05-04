# FundFlow AI: Technical Deep Dive 🕵️‍♂️📈

This document provides a detailed technical breakdown of FundFlow AI. It is designed to help you explain the "how" and "why" behind the project during your presentation.

---

## 1. The Problem Statement
Public Sector Banks (PSBs) in India move millions of transactions daily via UPI, IMPS, and NEFT. Standard fraud detection often:
1.  **Analyzes in isolation**: It misses the "web" of transactions.
2.  **Is too slow**: It flags fraud *after* the money has left the ecosystem.
3.  **Lacks context**: It doesn't factor in India-specific rails like KYC tiers or VPA reputation.

**Our Solution**: A real-time, graph-powered intelligence platform that uses a **Triple-Brain Architecture** to score transactions in milliseconds.

---

## 2. Theoretical Framework: The Triple-Brain System

We don't rely on just one model. We use three independent layers of intelligence to ensure maximum accuracy:

### 🧠 Brain 1: The Machine Learner (XGBoost)
*   **Engine**: Python (FastAPI) + XGBoost.
*   **Features**: 49 engineered features including velocity, amount deviations, and time-of-day anomalies.
*   **Performance**: Trained on 400k transactions with an **AUC-ROC of 0.9666**.

### ⚙️ Brain 2: The 6-Layer Risk Engine (Heuristics)
Located in `server/services/riskEngine.js`, this layer uses a custom formula:
`FinalScore = (0.7 × MaxLayer) + (0.3 × WeightedAvg)`
The 6 layers are:
1.  **Location**: Flags cross-state or high-risk bank pairs (e.g., Rural/Cooperative banks).
2.  **Channel**: Inherent risk of the rail (ATM is higher risk than Branch).
3.  **Behavioral**: Statistical deviation from the account holder's 30-day average.
4.  **ML Score**: The output from the XGBoost model.
5.  **Network Graph**: Membership in known fraud rings or chains.
6.  **Velocity**: Transaction frequency (e.g., 8+ transfers in 5 minutes).

### 🤖 Brain 3: The Reasoner (Gemini LLM)
*   **Role**: The "Tie-Breaker."
*   **Logic**: For transactions in the "Uncertain Zone" (score 0.35–0.75), we send the full context (KYC, Location, Graph) to Gemini. It provides a human-like verdict and reasoning.

---

## 3. Graph Intelligence (Graph Theory)
One of our biggest innovations is moving from **rows** to **nodes**.
*   **Ring Detection**: We use NetworkX to find cycles (Money Laundering). If Account A -> B -> C -> A, it's a 100% fraud signal.
*   **Mule Detection**: We calculate a **Passthrough Ratio**. If an account receives ₹1 Lac and sends ₹99,900 within 10 minutes, and has a low balance history, it's flagged as a Money Mule.

---

## 4. India Stack Integration
We built this for the Indian context:
*   **VPA Reputation**: Analysis of Virtual Payment Address (age and PSP risk).
*   **KYC Risk**: Accounts with `OTP_BASED` KYC are weighted with higher risk than `FULL_KYC`.
*   **PMLA Thresholds**: Automatic tracking of "Structuring" — where transfers are kept just below ₹50,000 to avoid regulatory reporting.

---

## 5. Technology Stack
*   **Frontend**: React 19, Tailwind CSS v4 (Modern, High-Performance UI).
*   **Backend**: Node.js (Express), Prisma ORM (Type-safe DB access).
*   **Intelligence**: Python (FastAPI), XGBoost, NetworkX.
*   **Database**: PostgreSQL 16.
*   **Real-time**: Socket.IO for sub-100ms alert broadcasting.

---

## 6. Business Value for PSBs
1.  **Reduced False Positives**: The 6-layer engine ensures we don't block legitimate high-value customers.
2.  **Automated Investigation**: The Network Graph allows investigators to see the "Ring" instantly, saving hours of manual Excel tracing.
3.  **Explainability**: SHAP values tell the analyst *exactly* why a transaction was flagged (e.g., "Flagged due to unusual location and low KYC status").
