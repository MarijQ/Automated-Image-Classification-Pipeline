# Automated Image Classification Pipeline for Fingernail Diseases

## Overview

This repository contains the code, methods, and analysis for the MSc dissertation: **"Exploring Web-Scraping Pipelines for Image Classification"**, completed at Brunel University in 2024. 

The project developed an **automated pipeline** integrating **web-scraping**, **image filtering**, and **deep learning-based classification** (using CNNs) to address challenges in constructing high-quality medical datasets. The test case focused on **fingernail diseases**, such as onychomycosis, demonstrating a pipeline capable of improving diagnostic models in under-researched medical domains with minimal manual intervention.

---

## Key Contributions

- **Web-Scraping**: Automated data collection from multiple search engines using Puppeteer, supporting license filtering, concurrency, and error-handling mechanisms.
- **Image Filtering**: Three filtering methods implemented:
  - Manual thresholds (F1)
  - Interpretable filters with Logistic Regression (F2)
  - Feature extraction using Random Forest (F3)
- **Classification**: Performance comparison across three CNN approaches:
  - Basic CNN
  - Optimised CNN
  - Transfer learning leveraging MobileNetV2
- **Integrated Pipeline**: End-to-end solution combining scraping, filtering, and classification with active learning enhancements.
- Achieved **89% classification accuracy** with transfer learning.

---

## Features

- **Data Gathering**:
  - Utilised Puppeteer with anti-bot mechanisms (e.g., random user agents, headless browsers).
  - Scraped over 10,000 images across search engines (Bing, DuckDuckGo).
  - Managed license constraints on public image use.
- **Image Filtering**:
  - Advanced metrics for relevance (e.g., text percentage, edge density, dominant colours, clarity).
  - Active learning for incremental improvements.
- **Model Training**:
  - Augmentation techniques applied (e.g., rotation, brightness adjustments, flips).
  - Utilised TensorFlow/Keras for CNN architectures.
  - Evaluated filtering and classification pipelines (P1: semi-automated, P2: fully automated, P3: manual curation).
- **Ethics**:
  - Complied with ethical policies for web data collection.
  - Avoided personally identifiable information from images.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. **Python Requirements** (Tested with Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```

3. **JavaScript Requirements**:
   Install [Node.js](https://nodejs.org/) (v14+) and Puppeteer:
   ```bash
   npm install
   ```

4. **Set Up**:
   - Ensure a suitable GPU for deep learning tasks if using CNNs.
   - Puppeteer may require additional setup for headless browsing (see Puppeteer documentation).

---

## Project Structure

```
├── Webscraping/
│   ├── v13/
│   │   ├── webscraping.js       # Main web-scraping script (Puppeteer-based)
│   │   ├── results.xlsx         # URLs with metadata
│   │   ├── stats_*.xlsx         # Benchmark results
│   └── README.md                # Scraper-specific details
├── Filtering/
│   ├── v7/
│   │   ├── filtering.py         # Implementation of filtering approaches (F1-F3)
│   │   ├── simulate_max_results.xlsx
│   │   ├── benchmark_results.xlsx
│   └── README.md
├── Classification/
│   ├── v6/
│   │   ├── classification.py    # CNN models (C1-C3) and training workflows
│   │   ├── benchmark_results.xlsx
│   └── README.md
├── MSc_Dissertation_Final_Submitted.pdf
├── requirements.txt             # Python dependencies
└── README.md                    # Main documentation
```

---

## Usage

### 1. Web-Scraping

Run the scraper to collect images based on keywords:
```bash
node Webscraping/v13/webscraping.js
```

Resulting outputs are stored in `/output/` as images and metadata (`results.xlsx`).

### 2. Image Filtering

Run filtering approaches (selection via GUI):
```bash
python Filtering/v7/filtering.py
```

Results:
- Filtered images outputted with scores.
- Metrics stored in `simulate_max_results.xlsx`.

### 3. Classification

Train and evaluate CNN models:
```bash
python Classification/v6/classification.py
```

Customization options available for architectures (models: `SimpleCNN`, `TunedCNN`, `TransferLearning`).

---

## Results and Benchmarks

| **Pipeline** | **Accuracy** | **Precision** | **Recall** | **Human Effort** | **Training Time (s)** |
|--------------|--------------|---------------|------------|-------------------|-----------------------|
| Semi-Automated (P1) | 89.17%       | 91.81%        | 86.00%     | ~2hrs             | 249.19                |
| Fully Automated (P2) | 85.33%       | 81.18%        | 92.00%     | None              | 117.98                |
| Manual Approach (P3)  | 84.33%       | 76.28%        | 99.67%     | ~50hrs            | 75.67                 |

---

## Key Findings

- **Transfer Learning** outperformed other CNN models in robustness and accuracy.
- **F3 Filtering** achieved the highest relevance detection (78%) with minimal manual work.
- The **semi-automated pipeline (P1)** provides an optimal balance of scalability and quality.

---

## Future Directions

- Expand dataset using additional data sources (e.g., medical image repositories).
- Implement semi-supervised and unsupervised filtering.
- Explore advanced architectures (e.g., Vision Transformers).

---

## Acknowledgements

Gratitude to my supervisor, Dr Zear Ibrahim, and the Brunel University Department of Computer Science.

---

For full details, refer to the [Dissertation PDF](./MSc_Dissertation_Final_Submitted.pdf). 🚀
