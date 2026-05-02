# ECE 228 Project Proposal: Task Requirements & Execution Summary

## 1. Task Requirements (Provided by User)

### 1.1 Course Context
*   **Course:** ECE 228 (Machine Learning for Physical Applications).
*   **Current Progress:** Week 5 (Transformers). The core topic of the project (Physics-Informed Neural Networks - PINNs) is scheduled for Week 8.
*   **Project Selected:** Option 4: `PINNs-Torch` (High-performance Physics-Informed Neural Networks framework).

### 1.2 Proposal Submission Rules (from Syllabus/Instructions)
*   **Format:** PDF document, maximum 2 pages.
*   **Required Sections:**
    1.  **Problem Background & Motivation:** Introduce the problem, explain its importance, and describe the motivation for studying it.
    2.  **Related Work:** Discuss relevant research areas organized into main categories (themes/methods) rather than listing papers individually. Explain how the project is situated within existing literature.
    3.  **High-level Methodology:** Present the main idea of the proposed method and briefly describe the approach to the problem.

### 1.3 Tone & Stylistic Constraints
*   **Perspective:** Must be written from the perspective of a student currently in Week 5.
*   **Tone:** "Naive" and exploratory. It must sound like a student curious about upcoming topics, not an expert who has already mastered them.
*   **Pronouns:** Must use "we" as the primary pronoun.
*   **AI Artifacts:** Must strictly avoid overly professional, grandiose, or typical "AI-generated" phrasing.

---

## 2. Execution Strategy & Content Design

To meet the above requirements, the proposal was drafted with the following strategic choices:

### 2.1 Problem Background & Motivation
*   **Strategy:** Ground the motivation in the current course progress.
*   **Execution:** Started by acknowledging what has been learned so far (CNNs, Transformers) and their limitation (lack of physical constraints). Then, explicitly mentioned looking ahead at the syllabus to Part 2 ("Machine Learning for Physical Applications") and expressing a desire to "get a head start" on Week 8's topic (PINNs). This justifies why Week 5 students are proposing a Week 8 topic without sounding overly arrogant.

### 2.2 Related Work
*   **Strategy:** Fulfill the requirement to categorize by themes, using simple, student-friendly logic.
*   **Execution:** Divided into two themes:
    *   *Theme 1: Standard Deep Learning vs. Physics-Based Models.* Contrasted data-heavy models with PINNs, mentioning Neural ODEs (Week 6) as a comparison point to show course awareness.
    *   *Theme 2: Making PINNs Actually Run Fast.* Addressed the practical aspect. Mentioned the original 2019 paper but quickly pivoted to the chosen `pinns-torch` repository, explaining that it was chosen because it runs faster on normal laptops (a very realistic student concern).

### 2.3 High-level Methodology
*   **Strategy:** Propose a highly feasible, low-risk plan that relies heavily on the chosen repository's built-in features.
*   **Execution:**
    *   *Setup:* Explicitly stated we will use the repository's provided Navier-Stokes example and its `MeshSampler` to avoid the nightmare of downloading/cleaning massive external datasets.
    *   *Training:* Briefly explained the FCN architecture and the combined loss function (Data Loss + PDE Loss).
    *   *Experiment:* Proposed a simple, achievable optimization: tweaking parameters like `n_train` (number of points) or network layers to observe the trade-off between training time and accuracy.
    *   *Risk Management (Fallback Plan):* Added a crucial "Fallback Plan" stating that if Navier-Stokes is too computationally heavy, the team will pivot to the simpler 1D Schrödinger equation (which has a provided Jupyter Notebook in the repo). This demonstrates excellent project management and guarantees a deliverable.

### 2.4 Formatting & Output
*   **Strategy:** Provide ready-to-submit formats.
*   **Execution:** Generated the content in Markdown (`proposal_draft.md`), converted it to a properly formatted LaTeX document (`proposal.tex`), and compiled it directly into a 2-page PDF (`proposal.pdf`) using Python (`markdown-pdf`).