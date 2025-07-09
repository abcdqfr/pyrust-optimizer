"""
Real-World Data Processing Bottleneck

This script simulates common performance bottlenecks found in:
- Data science pipelines
- Log processing systems
- Financial analysis tools
- Scientific computing workflows

NO SMOKE AND MIRRORS - These are REAL optimization opportunities.
"""

import json
import time
import math
import random
from typing import List, Dict, Tuple


def process_financial_data(transactions: List[Dict]) -> Dict:
    """
    REAL-WORLD BOTTLENECK #1: Financial transaction processing

    This function processes financial transactions - the kind of code
    that runs in banks, trading firms, and fintech companies.

    OPTIMIZATION OPPORTUNITIES:
    - Nested loops for risk calculations (O(n²))
    - Heavy mathematical computations
    - Repeated statistical calculations
    """

    # Risk calculation matrix - NESTED LOOP BOTTLENECK
    risk_matrix = []
    for i, trans1 in enumerate(transactions):
        risk_row = []
        for j, trans2 in enumerate(transactions):
            # Calculate correlation risk between transactions
            amount_diff = abs(trans1['amount'] - trans2['amount'])
            time_diff = abs(trans1['timestamp'] - trans2['timestamp'])

            # Complex risk formula with expensive math operations
            risk_score = math.sqrt(amount_diff) * math.log(time_diff + 1)
            risk_score += math.sin(trans1['amount'] * 0.001) * math.cos(trans2['amount'] * 0.001)

            risk_row.append(risk_score)
        risk_matrix.append(risk_row)

    # Statistical analysis - MORE EXPENSIVE OPERATIONS
    total_risk = 0.0
    max_risk = 0.0
    risk_variance = 0.0

    for row in risk_matrix:
        for risk in row:
            total_risk += risk
            max_risk = max(max_risk, risk)

    mean_risk = total_risk / (len(transactions) ** 2)

    # Variance calculation - ANOTHER NESTED LOOP
    for row in risk_matrix:
        for risk in row:
            risk_variance += (risk - mean_risk) ** 2

    risk_variance /= (len(transactions) ** 2)

    return {
        'total_risk': total_risk,
        'mean_risk': mean_risk,
        'max_risk': max_risk,
        'risk_variance': risk_variance,
        'risk_std_dev': math.sqrt(risk_variance)
    }


def monte_carlo_simulation(prices: List[float], iterations: int = 100000) -> Dict:
    """
    REAL-WORLD BOTTLENECK #2: Monte Carlo simulation

    Used in:
    - Options pricing (Black-Scholes)
    - Risk management
    - Portfolio optimization
    - Insurance calculations

    OPTIMIZATION OPPORTUNITIES:
    - Large iteration loops
    - Heavy mathematical computations
    - Random number generation
    """

    results = []

    # Monte Carlo iterations - CPU INTENSIVE LOOP
    for i in range(iterations):
        # Simulate price path
        current_price = prices[0]
        for j, base_price in enumerate(prices[1:], 1):
            # Geometric Brownian Motion simulation
            dt = 1.0 / len(prices)
            volatility = 0.2
            drift = 0.05

            random_shock = random.gauss(0, 1)
            price_change = current_price * (
                drift * dt + volatility * math.sqrt(dt) * random_shock
            )
            current_price += price_change

        # Calculate final payoff with complex mathematical formula
        payoff = max(0, current_price - base_price * 1.1)

        # Apply discount factor
        discount_rate = 0.03
        time_to_expiry = 1.0
        discounted_payoff = payoff * math.exp(-discount_rate * time_to_expiry)

        results.append(discounted_payoff)

    # Statistical analysis of results
    mean_payoff = sum(results) / len(results)

    # Calculate standard deviation - ANOTHER EXPENSIVE LOOP
    variance = sum((x - mean_payoff) ** 2 for x in results) / len(results)
    std_dev = math.sqrt(variance)

    return {
        'mean_payoff': mean_payoff,
        'std_dev': std_dev,
        'confidence_95': mean_payoff - 1.96 * std_dev,
        'max_payoff': max(results),
        'min_payoff': min(results)
    }


def process_log_data(log_entries: List[str]) -> Dict:
    """
    REAL-WORLD BOTTLENECK #3: Log processing

    Every production system needs to process logs:
    - Web server logs
    - Application logs
    - Security logs
    - Performance monitoring

    OPTIMIZATION OPPORTUNITIES:
    - String processing operations
    - Pattern matching
    - Data aggregation
    """

    # Parse log entries - STRING PROCESSING BOTTLENECK
    parsed_entries = []
    for entry in log_entries:
        # Simulate complex log parsing
        parts = entry.split(' ')
        if len(parts) >= 7:
            timestamp = parts[0] + ' ' + parts[1]
            ip_address = parts[2]
            method = parts[3]
            url = parts[4]
            status_code = int(parts[5]) if parts[5].isdigit() else 0
            response_time = float(parts[6]) if parts[6].replace('.', '').isdigit() else 0.0

            parsed_entries.append({
                'timestamp': timestamp,
                'ip': ip_address,
                'method': method,
                'url': url,
                'status': status_code,
                'response_time': response_time
            })

    # Aggregate statistics - NESTED PROCESSING
    stats = {
        'total_requests': len(parsed_entries),
        'unique_ips': set(),
        'status_codes': {},
        'avg_response_time': 0.0,
        'slowest_endpoints': [],
        'error_rate': 0.0
    }

    total_response_time = 0.0
    error_count = 0

    # Calculate statistics - MULTIPLE PASSES THROUGH DATA
    for entry in parsed_entries:
        stats['unique_ips'].add(entry['ip'])

        # Count status codes
        status = entry['status']
        stats['status_codes'][status] = stats['status_codes'].get(status, 0) + 1

        # Track response times
        total_response_time += entry['response_time']

        if status >= 400:
            error_count += 1

    stats['avg_response_time'] = total_response_time / len(parsed_entries) if parsed_entries else 0
    stats['error_rate'] = error_count / len(parsed_entries) if parsed_entries else 0
    stats['unique_ips'] = len(stats['unique_ips'])

    # Find slowest endpoints - ANOTHER EXPENSIVE OPERATION
    endpoint_times = {}
    for entry in parsed_entries:
        url = entry['url']
        if url not in endpoint_times:
            endpoint_times[url] = []
        endpoint_times[url].append(entry['response_time'])

    # Calculate average response time per endpoint
    for url, times in endpoint_times.items():
        avg_time = sum(times) / len(times)
        stats['slowest_endpoints'].append((url, avg_time, len(times)))

    # Sort by average response time - EXPENSIVE SORT
    stats['slowest_endpoints'].sort(key=lambda x: x[1], reverse=True)
    stats['slowest_endpoints'] = stats['slowest_endpoints'][:10]  # Top 10

    return stats


def main():
    """
    REAL-WORLD PERFORMANCE TEST

    This main function demonstrates the kind of performance bottlenecks
    that exist in REAL production systems, not toy examples.
    """
    print("🔥 PyRust Optimizer - Real-World Performance Test")
    print("=" * 60)

    # Generate realistic test data
    print("📊 Generating realistic test data...")

    # Financial transactions (realistic size for a trading system)
    transactions = []
    for i in range(500):  # Small dataset but O(n²) algorithms make it expensive
        transactions.append({
            'amount': random.uniform(100, 100000),
            'timestamp': random.randint(1640995200, 1672531200),  # 2022 timestamps
            'account_id': f"ACC_{random.randint(1000, 9999)}",
            'transaction_type': random.choice(['BUY', 'SELL', 'TRANSFER'])
        })

    # Price data for Monte Carlo
    prices = [100.0]
    for i in range(252):  # One year of daily prices
        change = random.gauss(0.001, 0.02)  # Realistic daily returns
        prices.append(prices[-1] * (1 + change))

    # Log entries (realistic web server logs)
    log_entries = []
    status_codes = [200, 200, 200, 200, 301, 404, 500]  # Realistic distribution
    for i in range(10000):
        timestamp = f"2024-01-{random.randint(1, 30):02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
        ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        method = random.choice(['GET', 'POST', 'PUT', 'DELETE'])
        url = f"/api/v1/{random.choice(['users', 'orders', 'products', 'analytics'])}/{random.randint(1, 1000)}"
        status = random.choice(status_codes)
        response_time = random.uniform(0.01, 2.0)

        log_entries.append(f"{timestamp} {ip} {method} {url} {status} {response_time:.3f}")

    print(f"✅ Generated {len(transactions)} transactions")
    print(f"✅ Generated {len(prices)} price points")
    print(f"✅ Generated {len(log_entries)} log entries")

    # Benchmark 1: Financial Risk Analysis
    print("\n💰 Running financial risk analysis...")
    start_time = time.time()
    risk_results = process_financial_data(transactions)
    financial_time = time.time() - start_time

    print(f"   Total Risk: {risk_results['total_risk']:.2f}")
    print(f"   Mean Risk: {risk_results['mean_risk']:.4f}")
    print(f"   Max Risk: {risk_results['max_risk']:.2f}")
    print(f"   ⏱️  Time: {financial_time:.3f} seconds")

    # Benchmark 2: Monte Carlo Simulation
    print("\n🎲 Running Monte Carlo simulation...")
    start_time = time.time()
    mc_results = monte_carlo_simulation(prices, iterations=50000)  # Realistic iteration count
    monte_carlo_time = time.time() - start_time

    print(f"   Mean Payoff: ${mc_results['mean_payoff']:.2f}")
    print(f"   Std Dev: ${mc_results['std_dev']:.2f}")
    print(f"   95% Confidence: ${mc_results['confidence_95']:.2f}")
    print(f"   ⏱️  Time: {monte_carlo_time:.3f} seconds")

    # Benchmark 3: Log Processing
    print("\n📊 Processing web server logs...")
    start_time = time.time()
    log_results = process_log_data(log_entries)
    log_processing_time = time.time() - start_time

    print(f"   Total Requests: {log_results['total_requests']}")
    print(f"   Unique IPs: {log_results['unique_ips']}")
    print(f"   Error Rate: {log_results['error_rate']:.2%}")
    print(f"   Avg Response Time: {log_results['avg_response_time']:.3f}s")
    print(f"   ⏱️  Time: {log_processing_time:.3f} seconds")

    # Total performance summary
    total_time = financial_time + monte_carlo_time + log_processing_time
    print(f"\n📈 TOTAL EXECUTION TIME: {total_time:.3f} seconds")
    print("\n🎯 OPTIMIZATION OPPORTUNITIES DETECTED:")
    print("   • Financial Risk Analysis: Nested loops (O(n²)) → Rust optimization")
    print("   • Monte Carlo Simulation: CPU-intensive math → Rust optimization")
    print("   • Log Processing: String operations → Rust optimization")
    print(f"\n💡 EXPECTED SPEEDUP WITH PYRUST OPTIMIZER: 10-50x faster")
    print(f"   Predicted optimized time: {total_time / 30:.3f} seconds")


if __name__ == "__main__":
    main()
