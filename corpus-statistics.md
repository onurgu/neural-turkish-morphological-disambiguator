

# Results

- What is the frequency distribution of tagsets 
(tag sequences which their roots are stripped)

- n_analyses histogram

```
$ cat data/train.merge.utf8 | awk '!/^(<\/?S)|(<DOC)|(<\/?TITLE)/ { print NF-1;} ' | histogram.py                                                                            
# NumSamples = 839908; Min = 1.00; Max = 24.00
# Mean = 1.860486; Variance = 1.514736; SD = 1.230746; Median 1.000000
# each ∎ represents a count of 9812
    1.0000 -     3.3000 [735920]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
    3.3000 -     5.6000 [ 90242]: ∎∎∎∎∎∎∎∎∎
    5.6000 -     7.9000 [  9325]: 
    7.9000 -    10.2000 [  3986]: 
   10.2000 -    12.5000 [   307]: 
   12.5000 -    14.8000 [    64]: 
   14.8000 -    17.1000 [    57]: 
   17.1000 -    19.4000 [     0]: 
   19.4000 -    21.7000 [     3]: 
   21.7000 -    24.0000 [     4]: 

```

```

$ cat data/train.merge.utf8 | awk '!/^(<\/?S)|(<DOC)|(<\/?TITLE)/ { counts[NF-1] += 1;} END { for (count in counts) { printf "n_analyses: %2d ==> %6d\n", count, counts[count];} } '
n_analyses:  1 ==> 440685
n_analyses:  2 ==> 231858
n_analyses:  3 ==>  63377
n_analyses:  4 ==>  78108
n_analyses:  5 ==>  12134
n_analyses:  6 ==>   8413
n_analyses:  7 ==>    912
n_analyses:  8 ==>   3570
n_analyses:  9 ==>    191
n_analyses: 10 ==>    225
n_analyses: 11 ==>     10
n_analyses: 12 ==>    297
n_analyses: 13 ==>     24
n_analyses: 14 ==>     40
n_analyses: 15 ==>      2
n_analyses: 16 ==>     55
n_analyses: 20 ==>      3
n_analyses: 24 ==>      4


```


- sentence counts

```
$ cat data/train.merge.utf8 | awk '/^<S>/ { sentence_count = 0; } !/^(<\/?S)|(<DOC)|(<\/?TITLE)/ { sentence_count += 1} /^<\/S>/ { print sentence_count }' | histogram.py -p -x 110
# NumSamples = 50673; Min = 0.00; Max = 110.00
# 340 values outside of min/max
# Mean = 16.527875; Variance = 535.256742; SD = 23.135616; Median 12.000000
# each ∎ represents a count of 330
    0.0000 -    11.0000 [ 24767]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎ (48.88%)
   11.0000 -    22.0000 [ 15438]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎ (30.47%)
   22.0000 -    33.0000 [  6168]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎ (12.17%)
   33.0000 -    44.0000 [  2082]: ∎∎∎∎∎∎ (4.11%)
   44.0000 -    55.0000 [   827]: ∎∎ (1.63%)
   55.0000 -    66.0000 [   429]: ∎ (0.85%)
   66.0000 -    77.0000 [   251]:  (0.50%)
   77.0000 -    88.0000 [   171]:  (0.34%)
   88.0000 -    99.0000 [   119]:  (0.23%)
   99.0000 -   110.0000 [    81]:  (0.16%)
```

```
[['+Noun+A3sg+P3sg+Nom\n+Noun+A3sg+Pnon+Acc', 16156],
 ['+Noun+A3sg+Pnon+Nom\n+Noun+Prop+A3sg+Pnon+Nom', 14178],
 ['+Adj\n+Adverb\n+Det\n+Num+Card', 13467],
 ['+Det\n+Pron+Demons+A3sg+Pnon+Nom', 9637],
 ['+Adj\n+Noun+A3sg+Pnon+Nom', 8166],
 ['+Noun+A3sg+P3sg+Nom\n+Noun^DB+Adj+Almost', 6501],
 ['+Noun+A3pl+P3pl+Nom\n+Noun+A3pl+P3sg+Nom\n+Noun+A3pl+Pnon+Acc\n+Noun+A3sg+P3pl+Nom',
  6089],
 ['+Noun+A3sg+P2sg+Loc\n+Noun+A3sg+P3sg+Loc', 5833],
 ['+Verb+Pos^DB+Adj+PastPart+P3sg\n+Verb+Pos^DB+Noun+PastPart+A3sg+P3sg+Nom',
  5483],
 ['+Conj\n+Verb+Pos+Imp+A2sg', 5349],
 ['+Adj\n+Adverb', 5337],
 ['+Noun+A3sg+P2sg+Nom\n+Noun+A3sg+Pnon+Gen', 5131],
 ['+Adj\n+Noun+Prop+A3sg+Pnon+Nom', 4228],
 ['+Noun+A3sg+P2sg+Nom\n+Noun+A3sg+Pnon+Gen\n+Postp+PCNom\n+Verb+Pos+Imp+A2pl',
  4198],
 ['+Noun+A3sg+P2sg+Dat\n+Noun+A3sg+P3sg+Dat', 3988],
 ['+Noun+A3sg+P2sg+Acc\n+Noun+A3sg+P3sg+Acc', 3955],
 ['+Noun+Prop+A3sg+P2sg+Nom\n+Noun+Prop+A3sg+Pnon+Gen', 3816],
 ['+Noun+A3pl+P2sg+Nom\n+Noun+A3pl+Pnon+Gen', 3794],
 ['+Adj\n+Noun+A3sg+Pnon+Nom\n+Noun+Prop+A3sg+Pnon+Nom', 3594],
 ['+Adverb\n+Noun+A3sg+Pnon+Nom', 3520]]

```

```bash
cat data/train.merge.utf8 | awk '{ for (i=2; i <= NF; i++) { n_elems = split($i, arr, "+"); printf "%s\t%s\t", $1, arr[1]; for (j = 2; j <= n_elems; j++) { printf "+%s", arr[j]; }; printf "\n";  }; print "";}' | python utils.py --command generate_corpus_statistics --output_dir stats_with_ambiguity_scores_train_merge --gold_data 1 --verbose 0
```


# Tur, 2003

```bash

$ cat dataset/tr.train | awk '/^$/ { print sentence_length ; sentence_length = 0 } !/^$/ { sentence_length += 1 }' | sort -n | histogram.py -x 83
# NumSamples = 32171; Min = 1.00; Max = 83.00
# 19 values outside of min/max
# Mean = 13.229244; Variance = 100.296178; SD = 10.014798; Median 11.000000
# each ∎ represents a count of 186
    1.0000 -     9.2000 [ 13994]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
    9.2000 -    17.4000 [  9745]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
   17.4000 -    25.6000 [  4993]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
   25.6000 -    33.8000 [  2099]: ∎∎∎∎∎∎∎∎∎∎∎
   33.8000 -    42.0000 [   901]: ∎∎∎∎
   42.0000 -    50.2000 [   249]: ∎
   50.2000 -    58.4000 [   103]: 
   58.4000 -    66.6000 [    30]: 
   66.6000 -    74.8000 [    19]: 
   74.8000 -    83.0000 [    19]: 


```