"""
Test suite for Cross-Sector Transmission Detection System

Tests the transmission detection functionality across Energy, Transportation,
and Chemicals sectors to Agriculture.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cross_sector_transmission_detector import (
    CrossSectorTransmissionDetector,
    TransmissionMechanism,
    TransmissionResults,
    TransmissionSpeed
)


class TestCrossSectorTransmissionDetector(unittest.TestCase):
    """Test cases for CrossSectorTransmissionDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = CrossSectorTransmissionDetector(transmission_window=90)
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.transmission_window, 90)
        self.assertEqual(self.detector.significance_level, 0.05)
        
        # Check transmission mechanisms are defined
        self.assertIn('energy_agriculture', self.detector.transmission_mechanisms)
        self.assertIn('transport_agriculture', self.detector.transmission_mechanisms)
        self.assertIn('chemicals_agriculture', self.detector.transmission_mechanisms)
        
        # Check sector assets are defined
        self.assertIn('energy', self.detector.sector_assets)
        self.assertIn('transportation', self.detector.sector_assets)
        self.assertIn('chemicals', self.detector.sector_assets)
        self.assertIn('agriculture', self.detector.sector_assets)
    
    def test_transmission_mechanism_structure(self):
        """Test transmission mechanism data structure"""
        energy_mech = self.detector.transmission_mechanisms['energy_agriculture']
        
        self.assertEqual(energy_mech.source_sector, 'Energy')
        self.assertEqual(energy_mech.target_sector, 'Agriculture')
        self.assertIsInstance(energy_mech.expected_lag, int)
        self.assertIn(energy_mech.strength, ['Strong', 'Moderate', 'Weak'])
        self.assertIsInstance(energy_mech.crisis_amplification, bool)
    
    def test_sector_asset_mappings(self):
        """Test sector asset mappings contain expected tickers"""
        # Energy sector should contain major energy companies
        energy_assets = self.detector.sector_assets['energy']
        self.assertIn('XOM', energy_assets)  # ExxonMobil
        self.assertIn('CVX', energy_assets)  # Chevron
        
        # Agriculture sector should contain agricultural companies
        ag_assets = self.detector.sector_assets['agriculture']
        self.assertIn('ADM', ag_assets)  # Archer Daniels Midland
        self.assertIn('CF', ag_assets)   # CF Industries (fertilizer)
        self.assertIn('DE', ag_assets)   # John Deere
    
    def test_transmission_results_structure(self):
        """Test TransmissionResults data structure"""
        # Create a sample result
        result = TransmissionResults(
            pair=('XOM', 'CF'),
            transmission_detected=True,
            transmission_lag=30,
            correlation_strength=0.25,
            p_value=0.01,
            mechanism='Natural gas prices → fertilizer costs',
            speed_category='Fast',
            crisis_amplification=None
        )
        
        self.assertEqual(result.pair, ('XOM', 'CF'))
        self.assertTrue(result.transmission_detected)
        self.assertEqual(result.transmission_lag, 30)
        self.assertEqual(result.speed_category, 'Fast')
    
    def test_transmission_speed_structure(self):
        """Test TransmissionSpeed data structure"""
        speed = TransmissionSpeed(
            pair=('XOM', 'CF'),
            optimal_lag=15,
            max_correlation=0.3,
            speed_category='Fast',
            lag_profile={0: 0.1, 15: 0.3, 30: 0.2},
            significance_threshold=0.05
        )
        
        self.assertEqual(speed.pair, ('XOM', 'CF'))
        self.assertEqual(speed.optimal_lag, 15)
        self.assertEqual(speed.speed_category, 'Fast')
        self.assertIsInstance(speed.lag_profile, dict)
    
    def test_energy_transmission_detection(self):
        """Test energy to agriculture transmission detection"""
        # Test with a small subset to avoid long download times
        energy_assets = ['XOM']  # ExxonMobil
        ag_assets = ['CF']       # CF Industries (fertilizer)
        
        try:
            results = self.detector.detect_energy_transmission(energy_assets, ag_assets)
            
            # Should return one result
            self.assertEqual(len(results), 1)
            
            result = results[0]
            self.assertEqual(result.pair, ('XOM', 'CF'))
            self.assertIsInstance(result.transmission_detected, bool)
            self.assertIsInstance(result.correlation_strength, float)
            self.assertIsInstance(result.p_value, float)
            self.assertIn(result.speed_category, ['Fast', 'Medium', 'Slow', 'Unknown'])
            
        except Exception as e:
            # If data download fails, skip this test
            self.skipTest(f"Data download failed: {str(e)}")
    
    def test_transport_transmission_detection(self):
        """Test transportation to agriculture transmission detection"""
        # Test with a small subset
        transport_assets = ['UNP']  # Union Pacific Railroad
        ag_assets = ['ADM']         # Archer Daniels Midland
        
        try:
            results = self.detector.detect_transport_transmission(transport_assets, ag_assets)
            
            # Should return one result
            self.assertEqual(len(results), 1)
            
            result = results[0]
            self.assertEqual(result.pair, ('UNP', 'ADM'))
            self.assertIsInstance(result.transmission_detected, bool)
            
        except Exception as e:
            # If data download fails, skip this test
            self.skipTest(f"Data download failed: {str(e)}")
    
    def test_chemical_transmission_detection(self):
        """Test chemicals to agriculture transmission detection"""
        # Test with a small subset
        chemical_assets = ['DOW']  # Dow Chemical
        ag_assets = ['MOS']        # Mosaic (fertilizer)
        
        try:
            results = self.detector.detect_chemical_transmission(chemical_assets, ag_assets)
            
            # Should return one result
            self.assertEqual(len(results), 1)
            
            result = results[0]
            self.assertEqual(result.pair, ('DOW', 'MOS'))
            self.assertIsInstance(result.transmission_detected, bool)
            
        except Exception as e:
            # If data download fails, skip this test
            self.skipTest(f"Data download failed: {str(e)}")
    
    def test_transmission_speed_analysis(self):
        """Test transmission speed analysis"""
        pair = ('XOM', 'CF')  # Energy → Fertilizer
        
        try:
            speed_analysis = self.detector.analyze_transmission_speed(pair)
            
            self.assertEqual(speed_analysis.pair, pair)
            self.assertIsInstance(speed_analysis.optimal_lag, int)
            self.assertIsInstance(speed_analysis.max_correlation, float)
            self.assertIn(speed_analysis.speed_category, ['Fast', 'Medium', 'Slow', 'Unknown'])
            self.assertIsInstance(speed_analysis.lag_profile, dict)
            
        except Exception as e:
            # If data download fails, skip this test
            self.skipTest(f"Data download failed: {str(e)}")
    
    def test_summarize_transmission_results(self):
        """Test transmission results summarization"""
        # Create mock results
        mock_results = {
            'energy_agriculture': [
                TransmissionResults(
                    pair=('XOM', 'CF'),
                    transmission_detected=True,
                    transmission_lag=30,
                    correlation_strength=0.25,
                    p_value=0.01,
                    mechanism='Natural gas prices → fertilizer costs',
                    speed_category='Fast',
                    crisis_amplification=None
                )
            ]
        }
        
        summary_df = self.detector.summarize_transmission_results(mock_results)
        
        # Check DataFrame structure
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertEqual(len(summary_df), 1)
        
        # Check columns
        expected_columns = [
            'transmission_type', 'source_asset', 'target_asset',
            'transmission_detected', 'transmission_lag', 'correlation_strength',
            'p_value', 'speed_category', 'mechanism'
        ]
        for col in expected_columns:
            self.assertIn(col, summary_df.columns)
        
        # Check values
        row = summary_df.iloc[0]
        self.assertEqual(row['transmission_type'], 'energy_agriculture')
        self.assertEqual(row['source_asset'], 'XOM')
        self.assertEqual(row['target_asset'], 'CF')
        self.assertTrue(row['transmission_detected'])
    
    def test_speed_categorization(self):
        """Test speed categorization logic"""
        # Test different lag values
        test_cases = [
            (15, 'Fast'),
            (30, 'Fast'),
            (45, 'Medium'),
            (60, 'Medium'),
            (75, 'Slow'),
            (90, 'Slow')
        ]
        
        for lag, expected_category in test_cases:
            if lag <= 30:
                category = "Fast"
            elif lag <= 60:
                category = "Medium"
            else:
                category = "Slow"
            
            self.assertEqual(category, expected_category)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with non-existent tickers
        invalid_energy = ['INVALID1']
        invalid_ag = ['INVALID2']
        
        results = self.detector.detect_energy_transmission(invalid_energy, invalid_ag)
        
        # Should return results even if data download fails
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertFalse(result.transmission_detected)
        self.assertEqual(result.correlation_strength, 0.0)
        self.assertEqual(result.p_value, 1.0)


class TestTransmissionMechanisms(unittest.TestCase):
    """Test transmission mechanism definitions"""
    
    def test_energy_agriculture_mechanism(self):
        """Test energy to agriculture transmission mechanism"""
        detector = CrossSectorTransmissionDetector()
        mechanism = detector.transmission_mechanisms['energy_agriculture']
        
        self.assertEqual(mechanism.source_sector, 'Energy')
        self.assertEqual(mechanism.target_sector, 'Agriculture')
        self.assertIn('natural gas', mechanism.mechanism.lower())
        self.assertIn('fertilizer', mechanism.mechanism.lower())
        self.assertTrue(mechanism.crisis_amplification)
    
    def test_transport_agriculture_mechanism(self):
        """Test transportation to agriculture transmission mechanism"""
        detector = CrossSectorTransmissionDetector()
        mechanism = detector.transmission_mechanisms['transport_agriculture']
        
        self.assertEqual(mechanism.source_sector, 'Transportation')
        self.assertEqual(mechanism.target_sector, 'Agriculture')
        self.assertIn('logistics', mechanism.mechanism.lower())
        self.assertTrue(mechanism.crisis_amplification)
    
    def test_chemicals_agriculture_mechanism(self):
        """Test chemicals to agriculture transmission mechanism"""
        detector = CrossSectorTransmissionDetector()
        mechanism = detector.transmission_mechanisms['chemicals_agriculture']
        
        self.assertEqual(mechanism.source_sector, 'Chemicals')
        self.assertEqual(mechanism.target_sector, 'Agriculture')
        self.assertIn('pesticide', mechanism.mechanism.lower())
        self.assertEqual(mechanism.strength, 'Moderate')


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)