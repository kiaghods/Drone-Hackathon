python3 from_txt.py -file tests_txt/couloir_avec_bureaux.txt -vis -nep -portions 0.25 0.25 0.25 0.25 -iter 10000 -show 0
La convergence ne semble pas aboutir, mince. Vérifier si je n'ai pas cassé DARP

python3 from_txt.py -file tests_txt/riviere.txt -vis -nep -portions 0.5 0.5 -iter 2000 -show 0.005
Il fait état d'un
```
corrupted size vs. prev_size
Abandon (core dumped)
```
après un certain nombre d'étapes
