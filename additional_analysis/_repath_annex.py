"""One-off: repoint appendix figure paths in annex.tex to the country-year-split
versions (combined predictions, per-country predictions, and SHAP). Verifies that
each old path occurs exactly once before substituting."""
import sys
ANNEX = "/Users/atin/Nowcasting/Nowcasting_Oxford_submission/annex.tex"
APP = "figures/Nowcast_Model_CYsplit/appendix"
SH  = "figures/Nowcast_Model_CYsplit/shap"
CC = ["US", "KR", "GB", "DE", "CA", "JP", "CN", "CH"]

mapping = {}
for C in ["AGT", "AGTwRD", "MGT", "MGTwRD", "Macros", "LagRD"]:
    mapping[f"figures/{C}/Test_Predictions_vs_TrueValues_Combined_log.pdf"] = f"{APP}/{C}_combined.png"
    mapping[f"figures/{C}/shap_summary_test.pdf"] = f"{SH}/shap_summary_{C}.png"
for C in ["AllVar", "AGT"]:
    for c in CC:
        mapping[f"figures/{C}/Test_Predictions_vs_TrueValues_{c}.pdf"] = f"{APP}/{C}_{c}.png"

txt = open(ANNEX).read()
bad = False
for old, new in mapping.items():
    n = txt.count(old)
    if n != 1:
        print(f"  [WARN] '{old}' occurs {n} times"); bad = True
if bad:
    print("Aborting due to unexpected counts."); sys.exit(1)
for old, new in mapping.items():
    txt = txt.replace(old, new)
open(ANNEX, "w").write(txt)
print(f"Replaced {len(mapping)} figure paths in annex.tex")
