# GSoC work
# What was done:
Worked on Exercise 5-1 BONUS: Adding GRU, LSTM, SimpleRNN support to the Keras parser.
## files changed
* `parser.py` — added import for `MakeKerasRNN`, uncommented the three lines in `mapKerasLayer`, fixed `return_sequences` logic in `MakeKerasRNN`
* `generate_keras_sequential.py` — added model generation for SimpleRNN, LSTM, GRU
* `generate_keras_functional.py` — same
* use the `bindings/pyroot/pythonization/test` folder to check the implemented functionality lib.
## Key findings:
* `rnn.py` was fully implemented but never wired into `parser.py`
* `return_sequences=False` fix: changed `nameY`/`nameY_h` assignment so final hidden state is used instead of full sequence output — fixed shape from `[1, 10, 1, 16]` to `[1, 1, 16]`
* The generated C++ `.hxx` reveals the actual computation is correct (`[1, 16]`), so it's purely a shape metadata mismatch

## findings during test runs 
* When doing the initial run just after uncommenting already implemented RNN, LSTM, GRU implementations, the error from the tests were because of shape issues.     
* the `parser.py` has a comment on line number 457 about this.      
* used that as reference and figured the extra dimension was `num_direction`.       
* the tests showed that SOFIE was generating the shapes correctly and allocating the desired memory, but the RModel metadata had issues.     
* to temporarily fix that, squeezed the extra dimension in `add_layer_into_RModel`

## after fix 
- the parsing now works for:
1. RNN
2. LSTM
3. GRU
- checked after building from `root_build` using the command 

```bash  
cmake --build . --parallel 4 --target install
```
```
```


```
# future work 
- will have to see `ROperator_RNN.hxx` to figure out why the metadata is not being stored the right way.
