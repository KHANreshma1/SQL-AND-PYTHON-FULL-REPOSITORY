# SQL-AND-PYTHON-FULL-REPOSITORY

1. SQL for Data Science (beyond Data Analyst)

👉 Tumne pehle practice kiya tha:

Top 3 highest paid employees

Department wise employee count

Name filtering etc.

Ab Data Scientist level ke liye aur advance queries:

A. Window Functions
-- Running total of sales by date
SELECT 
    order_date,
    SUM(sales) OVER (ORDER BY order_date) AS running_total
FROM orders;

-- Employee salary rank within department
SELECT 
    department_id,
    employee_name,
    salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_salary_rank
FROM employees;

B. Correlation check (SQL way)
-- Find correlation between age and salary (simplified)
SELECT 
    (COUNT(*)*SUM(age*salary) - SUM(age)*SUM(salary)) /
    (SQRT(COUNT(*)*SUM(age*age) - POWER(SUM(age),2)) *
     SQRT(COUNT(*)*SUM(salary*salary) - POWER(SUM(salary),2))) AS correlation
FROM employees;

C. Advanced Joins + Subqueries
-- Find employees earning more than department average
SELECT employee_name, salary, department_id
FROM employees e
WHERE salary > (
    SELECT AVG(salary) 
    FROM employees 
    WHERE department_id = e.department_id
);

2. Python for Data Science (beyond Analyst)

👉 Tumne pehle kiya tha:

Pandas for cleaning

Matplotlib/Seaborn for visualization

Basic statistics

Ab Data Scientist level ke liye advance modules:

A. Data Wrangling & Analysis
import pandas as pd

# Load data
df = pd.read_csv("employees.csv")

# Group by department and calculate mean salary
dept_salary = df.groupby("department")["salary"].mean()

# Detect outliers (IQR method)
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["salary"] < Q1 - 1.5*IQR) | (df["salary"] > Q3 + 1.5*IQR)]

B. Statistics + Hypothesis Testing
from scipy import stats

# Test if average salary of dept A and dept B are significantly different
deptA = df[df["department"]=="A"]["salary"]
deptB = df[df["department"]=="B"]["salary"]

t_stat, p_val = stats.ttest_ind(deptA, deptB)
print("t-stat:", t_stat, "p-value:", p_val)

C. Machine Learning Basics (sklearn)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[["age","experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("R2 Score:", model.score(X_test, y_test))

D. Visualization for Insights
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
