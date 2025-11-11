"""
Bayesian Analysis Module
Advanced statistical analysis for circuit lower bounds validation
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from statsmodels.stats.multitest import multipletests

class BayesianAnalyzer:
    """Bayesian analysis for hypothesis testing"""
    
    def __init__(self, prior_prob: float = 0.5):
        self.prior_prob = prior_prob
    
    def compute_bayesian_factors(self, validation_results: Dict) -> Dict:
        """
        Compute Bayes factors for all validation results
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Dictionary of Bayes factors and interpretations
        """
        bayesian_results = {}
        
        for key, result in validation_results.items():
            if 'independent' in key:
                # Compute Bayes factor for H0: circuit is independent
                bf = self._compute_bayes_factor(result)
                
                bayesian_results[key] = {
                    'bayes_factor': bf,
                    'evidence_strength': self._interpret_bayes_factor(bf),
                    'posterior_probability': self._compute_posterior(bf, self.prior_prob),
                    'log_bayes_factor': np.log10(bf) if bf > 0 else -np.inf
                }
        
        return bayesian_results
    
    def _compute_bayes_factor(self, result: Dict) -> float:
        """
        Compute Bayes factor using combined p-values
        
        Uses Fisher's method to combine p-values from multiple tests,
        then converts to Bayes factor using BIC approximation
        """
        # Extract p-values from all tests
        p_values = []
        
        test_names = ['cross_influence', 'concentration', 'product_bound', 
                     'kkl_condition', 'mcdiarmid_condition']
        
        for test_name in test_names:
            test_data = result.get(test_name, {})
            p_val = test_data.get('p_value', 1.0)
            # Avoid log(0)
            p_val = max(p_val, 1e-10)
            p_values.append(p_val)
        
        # Fisher's combined probability test
        chi_square = -2 * sum(np.log(p) for p in p_values)
        df = 2 * len(p_values)
        
        # Combined p-value
        combined_p = stats.chi2.sf(chi_square, df)
        combined_p = max(combined_p, 1e-10)
        
        # BIC approximation: BF ≈ exp(BIC/2)
        # For large n, BIC ≈ -2 * log(p_value)
        n = result.get('n', 100)  # Sample size proxy
        bic = -2 * np.log(combined_p)
        
        # Bayes factor for H1 vs H0
        bf = np.exp(bic / 2)
        
        # Cap at reasonable value
        return min(bf, 1e10)
    
    def _interpret_bayes_factor(self, bf: float) -> str:
        """Interpret Bayes factor using Jeffreys' scale"""
        if bf > 100:
            return "Decisive evidence for H1"
        elif bf > 30:
            return "Very strong evidence for H1"
        elif bf > 10:
            return "Strong evidence for H1"
        elif bf > 3:
            return "Substantial evidence for H1"
        elif bf > 1:
            return "Anecdotal evidence for H1"
        elif bf > 1/3:
            return "Anecdotal evidence for H0"
        elif bf > 1/10:
            return "Substantial evidence for H0"
        elif bf > 1/30:
            return "Strong evidence for H0"
        elif bf > 1/100:
            return "Very strong evidence for H0"
        else:
            return "Decisive evidence for H0"
    
    def _compute_posterior(self, bf: float, prior: float) -> float:
        """
        Compute posterior probability using Bayes' theorem
        
        P(H1|D) = (BF * prior) / (BF * prior + (1 - prior))
        """
        numerator = bf * prior
        denominator = bf * prior + (1 - prior)
        return numerator / denominator if denominator > 0 else 0.0
    
    def compute_false_discovery_rate(self, validation_results: Dict, 
                                    alpha: float = 0.05) -> Dict:
        """
        Apply Benjamini-Hochberg FDR control
        
        Args:
            validation_results: Dictionary of validation results
            alpha: Desired FDR level
            
        Returns:
            Dictionary with FDR-adjusted results
        """
        # Collect all p-values for independent circuits
        p_values = []
        keys = []
        
        for key, result in validation_results.items():
            if 'independent' in key:
                # Use minimum p-value across all tests (most conservative)
                test_p_values = []
                
                for test_name in ['cross_influence', 'concentration', 'product_bound',
                                'kkl_condition', 'mcdiarmid_condition']:
                    test_data = result.get(test_name, {})
                    test_p_values.append(test_data.get('p_value', 1.0))
                
                p_values.append(min(test_p_values))
                keys.append(key)
        
        if not p_values:
            return {}
        
        # Apply Benjamini-Hochberg procedure
        rejected, corrected_pvals, alphac_sidak, alphac_bonf = multipletests(
            p_values, alpha=alpha, method='fdr_bh'
        )
        
        fdr_results = {}
        for i, key in enumerate(keys):
            fdr_results[key] = {
                'original_p': float(p_values[i]),
                'corrected_p': float(corrected_pvals[i]),
                'rejected': bool(rejected[i]),
                'fdr_controlled': corrected_pvals[i] <= alpha,
                'significance_level': alpha
            }
        
        return fdr_results
    
    def compute_credible_intervals(self, validation_results: Dict, 
                                  credible_level: float = 0.95) -> Dict:
        """
        Compute Bayesian credible intervals for test statistics
        
        Uses bootstrap samples to estimate posterior distributions
        """
        credible_results = {}
        
        for key, result in validation_results.items():
            intervals = {}
            
            for test_name in ['cross_influence', 'product_bound']:
                test_data = result.get(test_name, {})
                estimate = test_data.get('estimate', 0)
                std = test_data.get('std', 0)
                
                if std > 0:
                    # Assume normal posterior (with uninformative prior)
                    # This is reasonable for large sample sizes
                    z_score = stats.norm.ppf((1 + credible_level) / 2)
                    
                    lower = estimate - z_score * std
                    upper = estimate + z_score * std
                    
                    intervals[test_name] = {
                        'estimate': float(estimate),
                        'credible_lower': float(lower),
                        'credible_upper': float(upper),
                        'credible_level': credible_level
                    }
            
            if intervals:
                credible_results[key] = intervals
        
        return credible_results


class PowerAnalyzer:
    """Statistical power analysis"""
    
    def __init__(self):
        self.alpha = 0.05
    
    def compute_power_analysis(self, validation_results: Dict) -> Dict:
        """
        Compute statistical power for detecting differences between
        independent and dependent circuits
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Dictionary with power analysis results
        """
        # Compare independent vs dependent circuits
        independent_results = {k: v for k, v in validation_results.items() 
                             if 'independent' in k}
        dependent_results = {k: v for k, v in validation_results.items() 
                           if 'dependent' in k}
        
        if not independent_results or not dependent_results:
            return {'error': 'Need both independent and dependent results'}
        
        # Compute effect sizes
        effect_sizes = []
        powers = []
        
        for ind_key in independent_results:
            # Find corresponding dependent result
            dep_key = ind_key.replace('independent', 'dependent')
            
            if dep_key in dependent_results:
                ind_result = independent_results[ind_key]
                dep_result = dependent_results[dep_key]
                
                # Compute effect size for each test
                for test_name in ['cross_influence', 'product_bound']:
                    effect_size = self._compute_cohens_d(
                        ind_result.get(test_name, {}),
                        dep_result.get(test_name, {})
                    )
                    
                    if effect_size is not None:
                        effect_sizes.append(effect_size)
                        
                        # Compute power for this effect size
                        n = ind_result.get('n', 100)
                        power = self._compute_power(effect_size, n, self.alpha)
                        powers.append(power)
        
        if not effect_sizes:
            return {'error': 'Could not compute effect sizes'}
        
        # Aggregate results
        avg_effect_size = np.mean(effect_sizes)
        avg_power = np.mean(powers)
        min_power = np.min(powers)
        
        # Determine adequacy
        if avg_power >= 0.8:
            adequacy = "Excellent (≥0.8)"
        elif avg_power >= 0.6:
            adequacy = "Adequate (≥0.6)"
        else:
            adequacy = "Low (<0.6)"
        
        return {
            'average_power': float(avg_power),
            'min_power': float(min_power),
            'effect_size': float(avg_effect_size),
            'effect_sizes': [float(es) for es in effect_sizes],
            'powers': [float(p) for p in powers],
            'adequacy': adequacy,
            'alpha': self.alpha
        }
    
    def _compute_cohens_d(self, test1: Dict, test2: Dict) -> float:
        """
        Compute Cohen's d effect size
        
        d = (mean1 - mean2) / pooled_std
        """
        mean1 = test1.get('estimate', 0)
        mean2 = test2.get('estimate', 0)
        
        # Estimate std from confidence intervals
        ci_lower1 = test1.get('ci_lower', mean1)
        ci_upper1 = test1.get('ci_upper', mean1)
        std1 = (ci_upper1 - ci_lower1) / 3.92  # For 95% CI
        
        ci_lower2 = test2.get('ci_lower', mean2)
        ci_upper2 = test2.get('ci_upper', mean2)
        std2 = (ci_upper2 - ci_lower2) / 3.92
        
        if std1 <= 0 or std2 <= 0:
            return None
        
        # Pooled standard deviation
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        
        if pooled_std == 0:
            return None
        
        # Cohen's d
        d = abs(mean1 - mean2) / pooled_std
        
        return d
    
    def _compute_power(self, effect_size: float, n: int, alpha: float) -> float:
        """
        Compute statistical power for two-sample t-test
        
        Uses non-central t-distribution
        """
        # Non-centrality parameter
        delta = effect_size * np.sqrt(n / 2)
        
        # Critical value for two-tailed test
        df = 2 * n - 2
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # Power = P(reject H0 | H1 is true)
        # = P(|T| > t_crit | delta)
        power = 1 - stats.nct.cdf(t_crit, df, delta) + stats.nct.cdf(-t_crit, df, delta)
        
        return min(power, 1.0)
    
    def compute_sample_size_recommendation(self, desired_power: float = 0.8,
                                         expected_effect_size: float = 0.8,
                                         alpha: float = 0.05) -> Dict:
        """
        Recommend sample size for desired power
        
        Args:
            desired_power: Target statistical power (default 0.8)
            expected_effect_size: Expected Cohen's d
            alpha: Significance level
            
        Returns:
            Dictionary with sample size recommendations
        """
        # Binary search for required n
        n_min = 10
        n_max = 10000
        
        while n_max - n_min > 1:
            n_mid = (n_min + n_max) // 2
            power = self._compute_power(expected_effect_size, n_mid, alpha)
            
            if power < desired_power:
                n_min = n_mid
            else:
                n_max = n_mid
        
        recommended_n = n_max
        actual_power = self._compute_power(expected_effect_size, recommended_n, alpha)
        
        return {
            'recommended_sample_size': recommended_n,
            'desired_power': desired_power,
            'actual_power': float(actual_power),
            'effect_size': expected_effect_size,
            'alpha': alpha
        }


# Example usage
if __name__ == "__main__":
    # Simulated validation results
    validation_results = {
        'n=20,t=3_independent': {
            'n': 20, 't': 3,
            'cross_influence': {'estimate': 0.05, 'p_value': 0.001, 'ci_lower': 0.03, 'ci_upper': 0.07},
            'concentration': {'estimate': 0.98, 'p_value': 0.0001},
            'product_bound': {'estimate': 0.97, 'p_value': 0.0001, 'ci_lower': 0.95, 'ci_upper': 0.99},
            'kkl_condition': {'estimate': 0.96, 'p_value': 0.001},
            'mcdiarmid_condition': {'estimate': 0.95, 'p_value': 0.002}
        },
        'n=20,t=3_dependent': {
            'n': 20, 't': 3,
            'cross_influence': {'estimate': 0.45, 'p_value': 0.8, 'ci_lower': 0.40, 'ci_upper': 0.50},
            'concentration': {'estimate': 0.75, 'p_value': 0.5},
            'product_bound': {'estimate': 0.62, 'p_value': 0.6, 'ci_lower': 0.58, 'ci_upper': 0.66},
            'kkl_condition': {'estimate': 0.70, 'p_value': 0.4},
            'mcdiarmid_condition': {'estimate': 0.68, 'p_value': 0.5}
        }
    }
    
    # Bayesian analysis
    print("=" * 60)
    print("BAYESIAN ANALYSIS")
    print("=" * 60)
    
    bayesian = BayesianAnalyzer()
    bf_results = bayesian.compute_bayesian_factors(validation_results)
    
    for key, result in bf_results.items():
        print(f"\n{key}:")
        print(f"  Bayes Factor: {result['bayes_factor']:.2e}")
        print(f"  Evidence: {result['evidence_strength']}")
        print(f"  Posterior Probability: {result['posterior_probability']:.4f}")
    
    # FDR control
    print("\n" + "=" * 60)
    print("FALSE DISCOVERY RATE CONTROL")
    print("=" * 60)
    
    fdr_results = bayesian.compute_false_discovery_rate(validation_results)
    
    for key, result in fdr_results.items():
        print(f"\n{key}:")
        print(f"  Original p: {result['original_p']:.6f}")
        print(f"  Corrected p: {result['corrected_p']:.6f}")
        print(f"  FDR Controlled: {result['fdr_controlled']}")
    
    # Power analysis
    print("\n" + "=" * 60)
    print("STATISTICAL POWER ANALYSIS")
    print("=" * 60)
    
    power = PowerAnalyzer()
    power_results = power.compute_power_analysis(validation_results)
    
    print(f"\nAverage Power: {power_results['average_power']:.3f}")
    print(f"Min Power: {power_results['min_power']:.3f}")
    print(f"Effect Size (Cohen's d): {power_results['effect_size']:.3f}")
    print(f"Adequacy: {power_results['adequacy']}")
    
    # Sample size recommendation
    print("\n" + "=" * 60)
    print("SAMPLE SIZE RECOMMENDATION")
    print("=" * 60)
    
    ss_rec = power.compute_sample_size_recommendation()
    print(f"\nRecommended n: {ss_rec['recommended_sample_size']}")
    print(f"For power: {ss_rec['desired_power']:.2f}")
    print(f"Actual power: {ss_rec['actual_power']:.3f}")
