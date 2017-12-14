
# coding: utf-8

# In[1]:


import pandas as pd
import sqlite3


# In[2]:


def new_connection(dbfile):
    
    """ Function to setup a new connection to a database (SQLite).
    
    Arguments:
    dbfile (char): Specify name of the database to be created
    
    Returns:
    Connection to database"""
    
    conn=sqlite3.connect(dbfile)
    return conn


# In[3]:


def type_col(col_type):
    
    """ Function to return the type of the column of a dataframe.
    
    Arguments:
    col_type (char): Datatype of the column in a dataframe
    
    Returns:
    Datatype to be created in SQL (this will be concatenated with the string)
    
    """
    if col_type=="int64":
        type1="INTEGER(64)"
    elif col_type=="float64":
        type1="FLOAT(64)"
    else:
        type1="VARCHAR(64)"
    return type1


# In[4]:


def generate_string(tablename, df, primarykey, foreignkey):
    
    """ Function to generate the SQL command as a string. This function can be used when only primary keys have to be created and there are no foreign keys associated.
    
    Arguments:
    tablename (char): Name of the table to be created in the database
    df (dataframe): Parsed dataframe output from which field names for the database will be extracted from the column names
    primarykey (char): Primary unique key associated with the table, this should be one of the columnnames from the dataframe
       
    
    Returns:
    An SQL command as a string which can be executed to create tables
    
    """
    
    columnnames=list(df.columns.values)
    columnnames=[item for item in columnnames if item not in foreignkey]
    
    sql_string="CREATE TABLE IF NOT EXISTS"+ " "+ tablename + "("
    
 
        
    for i in range(len(columnnames)):

        if i==len(columnnames)-1:
            col=columnnames[i]
            col_type=df[columnnames[i]].dtype

            if col==primarykey:
                coltype=type_col(col_type)
                sql_string = sql_string + col + " " + coltype + " " + "PRIMARY KEY" + " " + ")" + ";" 
            else:
                coltype=type_col(col_type)
                sql_string= sql_string + col + " " + coltype + " " + ")" + ";"

        else:

            col=columnnames[i]
            col_type=df[columnnames[i]].dtype

            if col==primarykey:
                coltype=type_col(col_type)
                sql_string = sql_string + col + " " + coltype + " " + "PRIMARY KEY" + ","

            else:
                coltype=type_col(col_type)
                sql_string= sql_string + col + " " + coltype + ","

    return (sql_string)


# In[5]:


def create_table(database, create_new_table_string):
    
    """ Function to create a new table in SQLite database.
     
    Arguments:
    database (char): Specify the name of the database in which the table has to be created
    create_new_table_string: SQL command to generate the table - spcify the name of table, fields (use generate_string function)
    
    Returns:
    The function creates a table in the specified database
    """
    
    connection = sqlite3.connect(database)
    cursor=connection.cursor()
    cursor.execute(create_new_table_string)
    connection.commit()
    connection.close()


# In[6]:


def push_dataframe(df,database, tablename):
    """ Function to push data in a dataframe into a table in the database.
    
    Arguments:
    df (dataframe): Dataframe which contains data to be pushed into tables
    database (char): Specify name of the database in which the table exists
    tablename (char): Name of the table in the specified database into which data has to be pushed 
    
    Returns:
    The function pushes data in the dataframe into the specified table
    """
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    df.to_sql(tablename,connection, if_exists="replace")
    connection.commit()
    connection.close()


# In[7]:


def query_data(database, query):
    """ Function to query data from the database.
    
    Arguments:
    database (char): Specify name of the database from which data has to be retrieved
    query (char): SQL command to query data
    
    Returns:
    The function returns the queried data in the dataframe format
    """
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    
    data=pd.read_sql(query,connection)
    connection.commit()
    connection.close()
    return data


# In[56]:


def foreignkey(tablename, foreignkey, foreigntable, foreignid):
    #col_type=originaldf[foreignkey].dtype
    #coltype=type_col(col_type)
    
    string = "ALTER TABLE" + " "+ tablename + " "+"add COLUMN" + " "+ foreignkey + " "+ "integer" + " " + "REFERENCES" + " " + foreigntable + "(" + foreignid + ")"
    return string


# In[13]:


def create_key(database, foreignstring):
    connection = sqlite3.connect(database)
    cursor=connection.cursor()
    cursor.execute(foreignstring)
    connection.commit()
    connection.close()

