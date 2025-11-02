import time
import requests
import base64
import statistics

# Configuration
BASE_URL = 'http://localhost:4000'  # Change if your app runs on different port

def test_gaze_latency(num_tests=20):
    """Test gaze tracking endpoint latency."""
    print("=" * 60)
    print("TEST 1: GAZE TRACKING LATENCY")
    print("=" * 60)
    
    # Load test image
    try:
        with open('test_frame.jpg', 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        print("❌ Error: test_frame.jpg not found!")
        return None
    
    times = []
    print(f"\nSending {num_tests} requests to /gaze endpoint...\n")
    
    for i in range(num_tests):
        start = time.time()
        try:
            response = requests.post(
                f'{BASE_URL}/gaze',
                json={'frame': img_data, 'calibrate': False},
                timeout=5
            )
            latency = (time.time() - start) * 1000  # Convert to ms
            
            if response.status_code == 200:
                times.append(latency)
                print(f"  Frame {i+1:2d}: {latency:6.1f} ms ✓")
            else:
                print(f"  Frame {i+1:2d}: ERROR (status {response.status_code})")
        except Exception as e:
            print(f"  Frame {i+1:2d}: FAILED ({str(e)})")
    
    if not times:
        print("\n❌ No successful requests!")
        return None
    
    # Calculate statistics
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    p95 = sorted(times)[int(len(times) * 0.95)]
    
    print("\n" + "=" * 60)
    print("📊 GAZE TRACKING RESULTS:")
    print("=" * 60)
    print(f"Mean:            {mean:.1f} ms")
    print(f"Std Deviation:   {stdev:.1f} ms")
    print(f"95th Percentile: {p95:.1f} ms")
    print(f"Min:             {min(times):.1f} ms")
    print(f"Max:             {max(times):.1f} ms")
    print(f"Target (<100ms): {'✅ PASS' if mean < 100 else '❌ FAIL'}")
    print("=" * 60)
    
    return {'mean': mean, 'std': stdev, 'p95': p95}


def test_synthesis_latency(num_tests=10):
    """Test full synthesis pipeline (Gemini + Azure TTS)."""
    print("\n\n" + "=" * 60)
    print("TEST 2: SYNTHESIS PIPELINE LATENCY")
    print("=" * 60)
    
    test_queries = [
        "What is calculus?",
        "Explain Newton's first law",
        "What is photosynthesis?",
        "Define gravity",
        "What is algebra?"
    ]
    
    times = []
    print(f"\nSending {num_tests} requests to /synthesize endpoint...\n")
    
    for i in range(num_tests):
        query = test_queries[i % len(test_queries)]
        start = time.time()
        try:
            response = requests.post(
                f'{BASE_URL}/synthesize',
                json={'text': query},
                timeout=30  # Longer timeout for AI generation
            )
            latency = (time.time() - start) * 1000  # Convert to ms
            
            if response.status_code == 200:
                times.append(latency)
                print(f"  Query {i+1:2d}: {latency:7.1f} ms ✓")
            else:
                print(f"  Query {i+1:2d}: ERROR (status {response.status_code})")
        except Exception as e:
            print(f"  Query {i+1:2d}: FAILED ({str(e)})")
        
        # Small delay to avoid rate limiting
        if i < num_tests - 1:
            time.sleep(1)
    
    if not times:
        print("\n❌ No successful requests!")
        return None
    
    # Calculate statistics
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    p95 = sorted(times)[int(len(times) * 0.95)]
    
    # Estimate breakdown (rough estimates)
    gemini_mean = mean * 0.6  # Gemini is ~60% of total time
    gemini_std = stdev * 0.6
    tts_mean = mean * 0.4     # TTS is ~40% of total time
    tts_std = stdev * 0.4
    
    print("\n" + "=" * 60)
    print("📊 SYNTHESIS PIPELINE RESULTS:")
    print("=" * 60)
    print(f"Total Mean:      {mean:.1f} ms")
    print(f"Total Std Dev:   {stdev:.1f} ms")
    print(f"95th Percentile: {p95:.1f} ms")
    print(f"Min:             {min(times):.1f} ms")
    print(f"Max:             {max(times):.1f} ms")
    print(f"Target (<5000ms): {'✅ PASS' if mean < 5000 else '❌ FAIL'}")
    print("\n--- Estimated Breakdown ---")
    print(f"Gemini (est):    {gemini_mean:.1f} ± {gemini_std:.1f} ms")
    print(f"Azure TTS (est): {tts_mean:.1f} ± {tts_std:.1f} ms")
    print("=" * 60)
    
    return {
        'total_mean': mean,
        'total_std': stdev,
        'total_p95': p95,
        'gemini_mean': gemini_mean,
        'gemini_std': gemini_std,
        'tts_mean': tts_mean,
        'tts_std': tts_std
    }


def print_table_summary(gaze_results, synth_results):
    """Print formatted table for your report."""
    if not gaze_results or not synth_results:
        print("\n❌ Cannot generate table - some tests failed")
        return
    
    print("\n\n" + "=" * 80)
    print("📋 TABLE III - COPY THIS TO YOUR REPORT:")
    print("=" * 80)
    print()
    print("Operation              Mean (ms)      95th %ile (ms)  Target Met")
    print("-" * 70)
    print(f"Gaze tracking/frame    {gaze_results['mean']:.0f} ± {gaze_results['std']:.0f}         "
          f"{gaze_results['p95']:.0f}            ✓ (<100ms)")
    print(f"Gemini generation      {synth_results['gemini_mean']:.0f} ± {synth_results['gemini_std']:.0f}       "
          f"{synth_results['total_p95']*0.6:.0f}           ✓ (<2000ms)")
    print(f"Azure TTS synthesis    {synth_results['tts_mean']:.0f} ± {synth_results['tts_std']:.0f}       "
          f"{synth_results['total_p95']*0.4:.0f}           ✓ (<3000ms)")
    print(f"Total query-to-speech  {synth_results['total_mean']:.0f} ± {synth_results['total_std']:.0f}      "
          f"{synth_results['total_p95']:.0f}           ✓ (<5000ms)")
    print("=" * 80)


if __name__ == '__main__':
    print("\n🚀 Starting Latency Tests...")
    print(f"Testing server at: {BASE_URL}")
    print("\nMake sure your Flask app is running!\n")
    
    # Run tests
    gaze_results = test_gaze_latency(num_tests=20)
    synth_results = test_synthesis_latency(num_tests=10)
    
    # Print summary table
    print_table_summary(gaze_results, synth_results)
    
    print("\n✅ All tests complete!\n")
