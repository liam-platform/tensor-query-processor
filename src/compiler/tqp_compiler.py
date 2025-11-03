from spark import SparkPhysicalPlanParser, SparkPhysicalPlan
from ir import IROptimizer
from planner import TQPPlanner
from executor import TQPExecutor


class TQPCompiler:
    """Full TQP compilation pipeline: Spark Plan -> IR -> Optimized IR -> Tensor Programs"""

    def __init__(self):
        self.parser = SparkPhysicalPlanParser()
        self.optimizer = IROptimizer()
        self.planner = TQPPlanner()

    def compile(self, spark_plan: SparkPhysicalPlan) -> TQPExecutor:
        """Execute full 4-layer compilation pipeline"""
        # Layer 1: Parse Spark physical plan to IR
        ir = self.parser.parse(spark_plan)

        # Layer 2: Canonicalize and optimize IR
        ir = self.optimizer.canonicalize(ir)
        ir = self.optimizer.optimize(ir)

        # Layer 3: Generate operator plan (tensor programs)
        plan = self.planner.plan(ir)

        # Layer 4: Create executor
        return TQPExecutor(plan)
