
# coding: utf-8

# In[1]:


from parsing import *
from database_functions import *
from ml_functions import *


# In[2]:


data_raw_fname = '/Users/jenniferpolson/Documents/School/2017-F/BE 223A/Final Project/Functionalize/be223a_dataset_full.csv'
encoding="ISO-8859-1"
dtformat='%Y-%m-%d %H:%M:%S'
patients, appointments, weather = parsing(data_raw_fname, encoding, dtformat, 
                                          exam_id = 'examId', pt_id = 'patient_deid', age = 'AgeAtExam', gender = 'Gender')


# In[3]:


data_raw_fname = '/Users/jenniferpolson/Documents/School/2017-F/BE 223A/Midterm Project/be223a_dataset.csv'
encoding=None
dtformat='%m/%d/%y %H:%M'
patients_s, appointments_s, weather_s = parsing(data_raw_fname, encoding, dtformat, 
                                          exam_id = 'Exam ID', pt_id = 'Patient ID', age = 'Age', gender = 'Gender')


# In[4]:


patients=patients.rename(columns = {'patient_deid':'Patient_ID'})
appointments=appointments.rename(columns = {'examId':'Exam_ID', 'patient_deid':'Patient_ID'})


# In[5]:


#Input Parameters
dbname = "223ADB3_v.db"
patient_key = "Patient_ID"
appointment_key = "Exam_ID"
weather_key = "Dayofyear"


# In[6]:


connection = new_connection(dbname)


# In[7]:


#Generating SQL Strings
patients_string=generate_string("patients", patients , patient_key, [])
appointments_string=generate_string("appointments", appointments, appointment_key, [patient_key, weather_key])
weather_string=generate_string("weather", weather, "Dayofyear", [])


# In[8]:


# Creating tables

create_table(dbname, patients_string)
create_table(dbname, appointments_string)
create_table(dbname, weather_string)


# In[9]:


# Creating foreign keys
fkey1=foreignkey("appointments", patient_key, "patients", patient_key)
fkey2=foreignkey("appointments", weather_key, "weather", weather_key)


# Creating foreign key columns
#create_key(dbname, fkey1)
#create_key(dbname, fkey2)


# In[10]:


# Push dataframe into database
push_dataframe(patients,dbname, "patients")
push_dataframe(appointments, dbname, "appointments")
push_dataframe(weather,dbname, "weather")


# In[11]:


#other function: pull from DB, rearrange to data output
patients = query_data(dbname, "SELECT * from patients") 
appointments = query_data(dbname, "SELECT * from appointments")
weather = query_data(dbname, "SELECT * from weather")


# In[12]:


df1 = create_df(appointments, patients, weather, 'Patient_ID', 'Dayofyear').drop('index', axis = 1)


# In[14]:


#run ML
results, results_v, test, validate_e, prob, groups, evalstats, sort_features = run_model(df1, appointments_s, k = 5000, tune_parameters = False)


# In[15]:


#create new table - test results
results_string = generate_string ('results', results, 'Patient_ID', []) #patient ID
create_table(dbname, results_string)

#push the data to the table
push_dataframe(results,dbname, 'results')


# In[16]:


#create new table - test results
results_v_string = generate_string ('vresults', results_v, 'Patient_ID', []) #patient ID
create_table(dbname, results_v_string)

#push the data to the table
push_dataframe(results_v,dbname, 'vresults')


# In[17]:


#circular - to make sure it actually worked
#results = query_data(dbname, "SELECT * from results") 

