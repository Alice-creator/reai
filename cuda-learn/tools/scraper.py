"""
tensortonic.com scraper — dùng Playwright để render JS rồi extract nội dung bài học.
Usage: python scraper.py [max_problems]
"""
import json, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

VENV_PYTHON = "/home/loc-dev/Downloads/venv/bin/python"
BASE_URL     = "https://www.tensortonic.com"
OUT_DIR      = Path(__file__).parent / "lessons"
OUT_DIR.mkdir(exist_ok=True)

# ── Danh sách bài từ bundle đã cào ──────────────────────────────────────────
PROBLEMS = [
    # (section, topic, slug, url_path)
    # Statistics
    ("Statistics", "Basic Statistics",            "calculate-mean",             "/ml-math/statistics/basic-statistics"),
    ("Statistics", "Basic Statistics",            "calculate-median",           "/ml-math/statistics/basic-statistics"),
    ("Statistics", "Basic Statistics",            "calculate-variance-std",     "/ml-math/statistics/basic-statistics"),
    ("Statistics", "Population vs Sample",        "population-sample-stats",    "/ml-math/statistics/population-sample"),
    ("Statistics", "Sampling Distributions",      "standard-error-calculation", "/ml-math/statistics/sampling-distribution"),
    ("Statistics", "Central Limit Theorem",       "clt-simulation",             "/ml-math/statistics/central-limit-theorem"),
    ("Statistics", "Confidence Intervals",        "ci-mean-known-sigma",        "/ml-math/statistics/confidence-interval"),
    ("Statistics", "Hypothesis Testing",          "hypothesis-setup",           "/ml-math/statistics/hypothesis-testing"),
    ("Statistics", "P-Values",                    "p-value-from-z",             "/ml-math/statistics/p-value"),
    ("Statistics", "T-Test",                      "t-test-statistic",           "/ml-math/statistics/t-test"),
    ("Statistics", "A/B Testing",                 "ab-test-setup",              "/ml-math/statistics/ab-testing"),
    ("Statistics", "Correlation",                 "pearson-correlation",        "/ml-math/statistics/correlation"),
    ("Statistics", "MLE",                         "mle-bernoulli",              "/ml-math/statistics/maximum-likelihood-estimation"),
    # Probability
    ("Probability", "Conditional Probability",    "conditional-probability",    "/ml-math/probability/conditional-probability"),
    ("Probability", "Distributions",              "pmf-pdf-cdf",                "/ml-math/probability/distributions"),
    ("Probability", "Random Variables",           "expected-value-variance",    "/ml-math/probability/random-variables"),
    ("Probability", "Bayes Theorem",              "bayes-theorem",              "/ml-math/probability/bayes-theorm"),
    ("Probability", "Monte Carlo",                "monte-carlo-pi",             "/ml-math/probability/monte-carlo"),
    # Linear Algebra
    ("Linear Algebra", "Vector Norms",            "vector-norms",               "/ml-math/linear-algebra/dot-product-norms"),
    ("Linear Algebra", "Matrix Multiplication",   "matrix-multiplication",      "/ml-math/linear-algebra/matrix-mult-tensors"),
    ("Linear Algebra", "Gram-Schmidt",            "gram-schmidt",               "/ml-math/linear-algebra/orthogonality"),
    ("Linear Algebra", "Eigenvalues",             "eigenvalue-analysis",        "/ml-math/linear-algebra/eigenvalues"),
    ("Linear Algebra", "SVD",                     "svd-decomposition",          "/ml-math/linear-algebra/svd"),
    ("Linear Algebra", "PCA",                     "pca-from-scratch",           "/ml-math/linear-algebra/pca"),
    # Calculus
    ("Calculus", "Numerical Limits",              "numerical-limits",           "/ml-math/calculus/limits"),
    ("Calculus", "Gradient",                      "gradient-computation",       "/ml-math/calculus/partial-derivatives"),
    ("Calculus", "Chain Rule",                    "chain-rule-backprop",        "/ml-math/calculus/chain-rule"),
    ("Calculus", "Hessian",                       "hessian-computation",        "/ml-math/calculus/jacobian-hessian"),
    ("Calculus", "Taylor Series",                 "taylor-approximation",       "/ml-math/calculus/taylor-series"),
    ("Calculus", "Backpropagation",               "manual-backprop",            "/ml-math/calculus/backpropagation"),
    # Optimization
    ("Optimization", "Convexity",                 "convexity-check",            "/ml-math/optimization/convex-non-convex"),
    ("Optimization", "SGD",                       "sgd-minibatch",              "/ml-math/optimization/sgd-variants"),
    ("Optimization", "Momentum",                  "momentum-optimizer",         "/ml-math/optimization/momentum"),
    ("Optimization", "Adam",                      "adam-implementation",        "/ml-math/optimization/adaptive-lr"),
    ("Optimization", "Regularization",            "l1-l2-regularization",       "/ml-math/optimization/regularization"),
    # Information Theory
    ("Info Theory", "Shannon Entropy",            "shannon-entropy",            "/ml-math/information-theory/entropy"),
    ("Info Theory", "Cross-Entropy",              "cross-entropy-implementation","/ml-math/information-theory/cross-entropy"),
    ("Info Theory", "KL Divergence",              "kl-divergence",              "/ml-math/information-theory/kl-divergence"),
    ("Info Theory", "Mutual Information",         "mutual-information",         "/ml-math/information-theory/mutual-information"),
    ("Info Theory", "Information Gain",           "information-gain",           "/ml-math/information-theory/information-gain"),
]

MAX = int(sys.argv[1]) if len(sys.argv) > 1 else len(PROBLEMS)

def clean(text):
    return re.sub(r'\s+', ' ', text).strip()

def scrape_problem(page, section, topic, slug, path):
    url = f"{BASE_URL}{path}?problem={slug}"
    print(f"  → {url}")

    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(2)  # wait for React hydration

        # Try to find problem content using various selectors
        content = {}

        # Title
        for sel in ["h1", "h2", "[data-testid='problem-title']", ".problem-title"]:
            el = page.query_selector(sel)
            if el:
                content["title"] = clean(el.inner_text())
                break

        # Description / problem statement
        for sel in ["[data-testid='problem-description']", ".problem-description",
                    ".prose", "article", "main p", ".markdown"]:
            els = page.query_selector_all(sel)
            if els:
                texts = [clean(e.inner_text()) for e in els if len(clean(e.inner_text())) > 30]
                if texts:
                    content["description"] = texts[0]
                    break

        # Get all visible text from main content area
        main_text = ""
        for sel in ["main", "#main-content", ".content", "article"]:
            el = page.query_selector(sel)
            if el:
                main_text = clean(el.inner_text())
                break

        if not main_text:
            main_text = clean(page.inner_text("body"))

        content["raw_text"]  = main_text[:3000]
        content["section"]   = section
        content["topic"]     = topic
        content["slug"]      = slug
        content["url"]       = url

        return content

    except Exception as e:
        print(f"    ERROR: {e}")
        return {"section": section, "topic": topic, "slug": slug, "url": url, "error": str(e)}

def main():
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx     = browser.new_context(viewport={"width": 1280, "height": 800})
        page    = ctx.new_page()

        problems = PROBLEMS[:MAX]
        print(f"Scraping {len(problems)} problems...\n")

        for i, (section, topic, slug, path) in enumerate(problems):
            print(f"[{i+1}/{len(problems)}] {section} / {slug}")
            data = scrape_problem(page, section, topic, slug, path)
            results.append(data)

            # Save per-file
            out = OUT_DIR / f"{slug}.json"
            out.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        browser.close()

    # Save all
    summary = OUT_DIR / "_all.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone. Saved {len(results)} problems to {OUT_DIR}/")
    print(f"Full dump: {summary}")

if __name__ == "__main__":
    main()
