## Blackjax examples
Use the handley-lab fork of blackjax:
```
pip install git+https://github.com/handley-lab/blackjax@nested_sampling
```


# GW150914
run this quickstart with
```
uv run python jim/example/GW150914_blackjax.py --outdir ./gw150914 --N test --num-repeats 1
```

where num_repeats is the key parameter to dial up if you are getting unreliable results, this is an integer multiplying the number of MCMC steps to take, 1 is about as fast as possible, 5 is more reliable, 10 would be massive overkill. Runtime should be proportional to this 

# injections
run bns injections with
```
uv run python jim/example/injections/injection_blackjax.py --outdir ns --N bns_inject_1 --seed 1
```

Again repeats is the key parameter to dial up if results seem unreliable

