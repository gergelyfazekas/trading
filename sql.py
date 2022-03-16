import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Abcdef01.",
    database='first_db'
)

mycursor = db.cursor()
# mycursor.execute("DESCRIBE first_table")
# mycursor.execute("CREATE DATABASE first_db")

# mycursor.execute("CREATE TABLE first_table (name VARCHAR(50), age INT, date DATE)")
# db.commit()
# mycursor.execute("INSERT INTO first_table (name, age, date) VALUES ('GergelyFazekas', 26,'2022-03-14')")
# db.commit()
mycursor.execute("SELECT * FROM first_table")

# for c in mycursor:
#     print(c)

print(mycursor.fetchall())