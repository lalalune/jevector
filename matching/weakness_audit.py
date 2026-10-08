"""Run current regressions for the historical findings. No model calls are required."""
import unittest
from matching.test_production import ProductionTests

def main():
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProductionTests))
 if not result.wasSuccessful():raise SystemExit(1)
 print('Current matching regressions passed. Historical evidence remains in runs/matching/weakness-audit.json.')
if __name__=='__main__':main()
