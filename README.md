# Chinese2English_Seq2Seq_Attention

A Sequence to Sequence with Attention Model for translate chinese sentence to english sentence.
Reference to the paper: [https://aclweb.org/anthology/D15-1166][PlDb]

### Requirement
Here are packages must be installed.
```sh
$ pip install tensorflow==2.0.0-rc0 
$ pip install jieba
```
### Usage

train, translate functions is provided in this project now.
- train: can train your own model by this command line.
    ```sh
    $ python3 run_nn.py train 
    ```
- transalte: after input this command line will show the input line for typing your single chinese sentence. Press Ctrl+C to leave. ( This project provides the model trained only 5 EPOCHS. )
    ```sh
    $ python3 run_nn.py translate
    
    > Please enter chinese sentence: 湯姆累了。
    
    湯姆累了。->tom tried . 
    ```

### Result
Here are some Ch2Eng results from my project.

| Chinese | English |
| ------ | ------ |
| 你好 | how you are ! |
| 湯姆累了。 | tom tried . |
| 祝我好運。 | i hope i am good luck . |
| 瑪麗笑了。 | mary smiled . |
| 今天天氣很好。 | it's a nice day . |
|抱歉，我總是遲到。| i'm sorry that i am late . |
| 醫生叫我要多休息。 | the doctor called me a while i should take a while . |
| 其實我只想知道事實。 | i wonder if i know the facts . |
| 紐西蘭是個美麗的國家。| witzerland is a beautiful country . |
| 我想要養一隻貓，但我媽媽不讓我養。 | i want to the cat and i don't have a banana drinking . |