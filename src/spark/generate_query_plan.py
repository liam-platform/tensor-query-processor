import os
from pyspark.sql import SparkSession


class PhysicalQueryPlan:
    def __init__(self) -> None:
        self.sc = self.__initialize_spark()

    def __initialize_spark(self):
        """Initialize Spark session with proper error handling."""
        try:
            spark = (
                SparkSession
                .builder
                .appName("QueryPlanGenerator")
                .master("local[*]")   # optional, only for local testing
                .getOrCreate()
            )
            return spark
        except Exception as e:
            print(f"Error initializing Spark: {e}")
            raise

    def get_query_plan(self, sql_query, plan_type="extended"):
        """
        Get the query plan for a SQL query using Spark.
        
        Parameters:
        -----------
        sql_query : str
            The SQL query to analyze
        plan_type : str
            Type of plan to display:
            - "simple": Basic physical plan
            - "extended": Detailed plan with logical and physical plans
            - "cost": Plan with cost information
            - "formatted": Formatted plan (Spark 3.0+)
        
        Returns:
        --------
        str: The query plan
        """    
        try:
            # Create a DataFrame from the SQL query
            df = self.sc.sql(sql_query)
            
            # Get the query plan based on type
            if plan_type == "simple":
                plan = df._jdf.queryExecution().simpleString()
            elif plan_type == "extended":
                plan = df._jdf.queryExecution().toString()
            elif plan_type == "cost":
                plan = df._jdf.queryExecution().stringWithStats()
            elif plan_type == "formatted":
                plan = df._jdf.queryExecution().explainString(
                    self.sc._jvm.org.apache.spark.sql.execution.ExplainMode.fromString("formatted")
                )
            else:
                # Default to explain output
                plan = df._jdf.queryExecution().toString()
            
            return plan
        
        except Exception as e:
            return f"Error generating query plan: {str(e)}"
        
        finally:
            # Stop the Spark session
            self.sc.stop()
            pass


    def explain_query(self, sql_query, mode="extended"):
        """
        Alternative method using DataFrame.explain() which is more user-friendly.
        
        Parameters:
        -----------
        sql_query : str
            The SQL query to analyze
        mode : str
            Explain mode: "simple", "extended", "codegen", "cost", "formatted"
        """    
        try:
            df = self.sc.sql(sql_query)
            
            # Capture explain output
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            df.explain(mode=mode)
            
            sys.stdout = old_stdout
            plan = captured_output.getvalue()
            
            return plan
        
        except Exception as e:
            return f"Error generating query plan: {str(e)}"


# Example usage
# if __name__ == "__main__":
#     # Initialize Spark session
#     # Set environment variables for compatibility
#     os.environ.setdefault('PYSPARK_PYTHON', 'python3')
#     os.environ.setdefault('PYSPARK_DRIVER_PYTHON', 'python3')

#     spark = initialize_spark()
        
#     # Create sample data
#     data = [
#         (1, "Alice", 30, "Engineering"),
#         (2, "Bob", 35, "Sales"),
#         (3, "Charlie", 28, "Engineering"),
#         (4, "Diana", 32, "Marketing"),
#         (5, "Eve", 29, "Sales")
#     ]
    
#     df = spark.createDataFrame(data, ["id", "name", "age", "department"])
#     df.createOrReplaceTempView("employees")
    
#     # Example SQL query
#     sql_query = """
#         SELECT department, AVG(age) as avg_age, COUNT(*) as count
#         FROM employees
#         WHERE age > 28
#         GROUP BY department
#         ORDER BY avg_age DESC
#     """
    
#     print("=" * 80)
#     print("SQL Query:")
#     print("=" * 80)
#     print(sql_query)
#     print("\n")
    
#     # Method 1: Using get_query_plan
#     print("=" * 80)
#     print("Query Plan (Extended):")
#     print("=" * 80)
#     plan = get_query_plan(spark, sql_query, plan_type="extended")
#     print(plan)
#     print("\n")
    
#     # Method 2: Using explain_queryinte
#     print("=" * 80)
#     print("Query Plan (Using explain):")
#     print("=" * 80)
#     plan = explain_query(spark, sql_query, mode="extended")
#     print(plan)
    
#     # Stop Spark session
#     spark.stop()
