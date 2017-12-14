# BE223A
## Parsing

- David: Please see Parsing.ipynb to see an example of parsing. 
  - Overall parsing of the raw appointment data, and weather data (parsing.py). 
  - Integration and documentation of the ICD9 parsing functoinality that Talia provided. 
  - Preparation of the same set of features, folds, and training and test data for our midterm projects. 
  - Git repository management

- Talia: 
  - Traffic research
  - ICD9 background research and grouping by umbrella term
  - ICD9_archive folder
  - parsing_icd9.py
  - Reviewed Harvey et al. paper and presented key findings
  - Largely created group powerpoint presentation (i.e. future direction slides) and integrated slides

## Database

- Lavanya: Explored different DBMS options for implementation of Database module for the project  
  - Wrote all functions in database_functions.py (to setup connection, create tables, push / query data, reorganizing dataframes)
  - Code outline to integrate Web Application with Database
  - Co-wrote application_script.ipynb with Jennifer

## Machine learning

- Jen: wrote all functions in ml_functions.py (see documentation) to carry out the functionality of the model using queries from the database
  - Co-wrote Application_Script.ipynb with Lavanya, which served to integrate the parsing, database, and machine learning components
  - Helped write icd9_group function in parsing_icd9
  - Set up meetings and enforced deadlines to help the project move forward

## Web interface

- Akshaya: 
  - Python function for creating Dictionary for creating the dynamic forms
  - Python function for displaying Charts for displaying show/no show for all modalities and Hospital
  - Integration between Database and the Web application.
  - Also created dropdown features(html and css) but was not used in the final version of the web App.

- Chrysostomos:
  - Design of the layout of the dashboard.
  - Creation of hospital_map for the web form of the dashboard (./hospital_map.ipynb).
  - Construction of the dashboard form and calendar.
  - Help in the integration with the database.
  - Modification of the database with extra columns (for use in the web app).

- James:
  - Made bone structure for the web application.
  - Discussed with Chrysostomos and nailed down to the final layout.
  - Explored dropdown menu but the final version was implemented by the Chrysostomos.
  - Implemented interactive stacked bar chart and line chart using `Chart.js` template.
  - Retrieve data from the database for making charts.


