# Project report

`BING_Report_Group10.pdf` is the final report (IEEE-style, two columns). It is generated from the
results files, so every number matches the experiments:

```bash
python report/build_report.py      # writes report/report.html from results/*.json
# print to PDF with any Chromium-based browser, for example:
chromium --headless --no-pdf-header-footer --print-to-pdf=report/BING_Report_Group10.pdf report/report.html
```

Opening `report.html` in Chrome or Edge and using Print → Save as PDF (margins: default, no
headers and footers) gives the same result.
