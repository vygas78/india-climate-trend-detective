EARTHTREND EXPLORER
===================
A Windows desktop app (Python + Tkinter) for exploring environmental trends
across 28 Indian states, using two real CSV files:

  yearly_series.csv  - state, variable, year, value   (1,176 rows, 2003-2023)
  state_trends.csv   - state, variable, trend, change_per_year, p_value,
                       significant                     (56 rows)

Variables: T2M (temperature) and PRECTOTCORR (precipitation).


QUICK START (Windows 11)
------------------------
1. Right-click the ZIP file and choose "Extract All..." (do not run the app
   from inside the ZIP).
2. Open the extracted folder and double-click  run_windows.bat
3. The first time, it installs the needed packages (internet required, about
   1-3 minutes). After that the app opens straight away.

If Windows shows a blue "Windows protected your PC" box, click "More info"
and then "Run anyway".

Manual alternative: open Command Prompt in the folder and run
    py -m pip install -r requirements.txt
    py earthtrend_explorer.py


USING THE APP
-------------
The two CSV files are loaded automatically when they sit in the same folder
as earthtrend_explorer.py. To use other files, click "Load yearly CSV..." or
"Load trends CSV..." at the top. Use the two buttons at the top right to
switch pages.

PAGE 1 - Year-wise State Analysis
  - Choose a state, a variable, a start year and an end year.
  - The chart shows the actual yearly values plus the regression line and the
    Sen's slope line.
  - The panel on the right shows, CALCULATED from the yearly values:
      * Linear regression: slope (rate of change per year), 95% CI, p-value,
        R-squared
      * Kendall tau and its p-value
      * Sen's slope and its 95% confidence interval
      * A rough check of year-to-year correlation in the residuals
  - At least 5 years must be selected.
  - The "Supplied summary" part of the panel shows the matching row from
    state_trends.csv for comparison. Those numbers are NOT calculated by the
    app.

PAGE 2 - All-State Trend Overview
  - Bar chart and table of the rate of change (change_per_year) for all 28
    states, for temperature, precipitation, or both.
  - Orange bars = marked significant and increasing; blue bars = marked
    significant and decreasing; grey = not significant.
  - Everything on this page comes from the supplied state_trends.csv. The app
    does not recompute it. Sort the list or tick "Show only significant".
  - The toolbar under each chart lets you zoom and save the chart as an image.


WEB VERSION (optional, Streamlit)
---------------------------------
streamlit_app.py is a browser version of the same app. It uses the same
statistics and charts as the desktop app, so the numbers are identical.

Run it on your own computer:
    py -m pip install -r requirements.txt
    py -m streamlit run streamlit_app.py

Put it online for free (Streamlit Community Cloud):
  1. Keep all files in a public GitHub repository.
  2. Go to share.streamlit.io and sign in with GitHub.
  3. Click "Create app", choose the repository and branch, and set the main
     file path to streamlit_app.py.
  4. Click Deploy. You get a link anyone can open in a browser.
Free apps go to sleep when unused and take about 30 seconds to wake up.


READING THE NUMBERS CAREFULLY
-----------------------------
- Page 1 (calculated here) and Page 2 (supplied) can differ. In the supplied
  file, change_per_year equals the ordinary regression slope calculated on
  Page 1 for all 56 rows, but its p-values and significance flags come from a
  method that the file does not document. They do not match the regression
  p-value on Page 1 and are close to, but not identical with, the Kendall
  p-value.
- Regression, Kendall tau and Sen's slope treat each yearly value as
  independent. They do not account for serial correlation (a warm year tending
  to follow a warm year) or seasonality. This can make p-values look more
  convincing than they are. With 21 years of data, treat them as a rough
  guide, not proof.
- state_trends.csv contains 56 tests. At a 5% level, a few "significant"
  results are expected by chance alone.
- The CSV files give no measurement units and do not say how values were
  aggregated over each state, so this app does not describe them as state-wide
  averages. Some values (for example Himachal Pradesh and Sikkim temperatures
  close to or below typical lowland values) suggest the data may represent
  particular locations or grid cells; this has not been verified.
- A statistically significant trend over 2003-2023 is not by itself evidence
  of long-term climate change.


TROUBLESHOOTING
---------------
- "Python was not found": install Python from https://www.python.org/downloads/
  and tick "Add python.exe to PATH" in the installer.
- "Tkinter is missing": re-run the Python installer, choose Modify, and tick
  "tcl/tk and IDLE".
- "Package installation failed": check your internet connection, then run
  run_windows.bat again.
- "This does not look like the ... file": the CSV you picked is missing
  required columns. The message lists the columns that are needed.
- Files not loaded automatically: make sure yearly_series.csv and
  state_trends.csv are in the same folder as earthtrend_explorer.py, or load
  them with the buttons at the top.


FILES
-----
earthtrend_explorer.py   the application
yearly_series.csv        yearly data (real data supplied by the team)
state_trends.csv         pre-calculated trend summary (supplied)
requirements.txt         Python packages (pandas, numpy, scipy, matplotlib)
run_windows.bat          one-click launcher
streamlit_app.py         optional web version (Streamlit)
README.txt               this file
