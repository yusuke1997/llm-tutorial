# LLMチュートリアル@UEC

LLMの使い方に関する講座です。

>  [!NOTE]
>
> LLMにも種類がありますが、ここでは基本的に、CausalLM（GPTやLlamaのように文を逐次的に生成するモデル）を念頭に説明しています。BERTなどのencoderモデルや、mamba, LLaDAのようにTransformer系以外のモデルには、特殊な処理や関数、パッケージのインストールが必要になるかもしれませんが、一般的にLLMと呼ばれるモデルの大半はサポートできていると思います。



## 0.1 環境設定

サーバなど中央集約型のシステムにログインして作業するとき、それぞれのユーザごとに環境を作成する必要があります。

今回は環境を統一するために[uv](https://github.com/astral-sh/uv)というパッケージ管理ツールを使います。

サーバにログインした後、下記コマンドを実行してください。
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

その後に、 `uv -h`と入力したとき、
```bash
pine11% uv -h
An extremely fast Python package manager.

Usage: uv [OPTIONS] <COMMAND>
...
```
などのように出力されれば、インストール完了です。

---


## 0.2 仮想環境

各研究やプロジェクトごとに、異なるバージョンのライブラリを使用したいニーズが頻繁に発生します。
あるいは、本番環境と開発環境を分けたい状況等では、他のプロジェクトに影響を与えないように、
新しく環境を構築したいです。このような場面で、実際の自分の環境とは別に、プロジェクトごとの仮想環境を作成することが一般的です。

仮想環境の設定はuvを使った場合、非常に簡単です。下記のコードを実行してください。今回はPython3.10に統一しています。
```bash
uv venv --python 3.10
source .venv/bin/activate
```

このとき、
```bash
pine11% uv venv --python 3.10
source .venv/bin/activate
Using CPython 3.10.16
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
(llm-tutorial) pine11%
```
のように、括弧がカーソルの前に登場したら成功です。

試しに、`python`と入力すると、
```bash
(llm-tutorial) pine11% python
Python 3.10.16 (main, Dec  6 2024, 19:59:16) [Clang 18.1.8 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```
のようになって、`Python 3.10.16`がインストールされていることが確認できるはずです。

> [!NOTE]
>
> `source .venv/bin/activate`で仮想環境モードになっています。終了するときはシンプルにサーバとの接続を切るか、`deactivate`とコマンドを打つことで、先頭の括弧が消え、仮想環境から離れます。



### 参考資料

- https://docs.astral.sh/uv/
- https://zenn.dev/watanko/articles/8400df75e4aab5
- https://dev.classmethod.jp/articles/uv-unified-python-packaging-explained/
- https://qiita.com/yuya-0405/items/41c8153530d5542bee16
- https://zenn.dev/tabayashi/articles/52389e0d6c353a

---

## 1. Transformersの使い方

ローカルLLMを動かしていこうと思います。 [HuggingFace](https://huggingface.co/models) で大半のLLMが公開されているので、はじめに正式にサポートしている[Transformers](https://github.com/huggingface/transformers) ([Wolf et al., 2020](https://aclanthology.org/2020.emnlp-demos.6/)) を使用します。
ここからの解説は便宜上、すべて、単一のpythonコードへの記述を念頭に置いていますが、Jupyter Notebookなど、使いやすい形のものを使っていただいても構いませんが、サーバとJupyter Notebookの接続方法についてはここでは紹介しないので、各自調べてください。

> [!NOTE]
>
> Transformersを使用することが一般的ですが、通常のコンピュータで実行するには2つの課題があります。速度とメモリです。そのため計算方法を効率化した[vLLMs](https://github.com/vllm-project/vllm) ([Kwon et al., 2023](https://dl.acm.org/doi/10.1145/3600006.3613165)) 等を使うことで、推論速度を高速化したり、BitsAndBytes ([Dettmers et al., 2022](https://openreview.net/forum?id=dXiGWqBoxaD)) 等の量子化技術でメモリ削減する手法が開発されています。これらについては、後々詳細な説明を行います。

はじめに、下記コマンドでtransformersをinstallしてください。注意点として、`uv` を環境構築に使用しているため、`uv pip` のように`uv` を先頭に記述してください。`uv` を使用すると、installが高速になります。

```bash
uv pip install transformers
uv pip install "transformers[ja]"
uv pip install "transformers[sentencepiece]"
uv pip install torch
```

これで、Transformersのinstallは完了です。

また、以下のライブラリも後半で使うかもしれませんので、インストールしておいてください。

```bash
uv pip install vllm
uv pip install streamlit
uv pip install sacrebleu
```

念の為、GPUが使えることを以下のコマンドで確認してください。`True`になっていたらGPU環境で実行可能です。

```bash
python -c "import torch; print(torch.cuda.is_available())"
# True
```

---

### 補足：HuggingFaceのアカウント作成

[HuggingFace](https://huggingface.co/models) はほとんどのLLMが格納されているサービスです。大半のモデルはログインしなくても使用可能ですが、Llamaなど有名な一部のLLMは、ログインして使用申請（大半はチェックボックスのみで瞬時に利用可能）をしないと利用できません。そのため、HuggingFaceのアカウントを作成することをおすすめします。

https://huggingface.co/join

アカウントを作成したら、アクセストークンを発行します。これを使用することでサーバでもHuggingFaceにログインできるようになります。下記のURLにアクセスして、右上あたりにある `+ Create new token` ボタンを押せば発行できます。Permissionは最近追加されたもので、基本どれでもいいですが、`Write` が以前のデフォルトでしたので、私はこれを使用しています。

https://huggingface.co/settings/tokens

発行されたトークンは1度しか表示されないので、忘れずにコピペしておいてください。忘れると作り直しになります。

その後、CLI上で `huggingface-cli login`と入力してください。すると、以下のようにトークン情報を聞かれるので、先ほどコピペしたトークンを入力して、その後、保存するか聞かれるので、Yを押してください。以下のように`Login successful`が表示されたら成功です。

```bash
yusuke@mm-dhcp-124-238 llm-tutorial % huggingface-cli login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) Y
Token is valid (permission: write).
The token `test-for-uec` has been saved to /Users/yusuke/.cache/huggingface/stored_tokens
Your token has been saved in your configured git credential helpers (osxkeychain).
Your token has been saved to /Users/yusuke/.cache/huggingface/token
Login successful.
The current active token is: `test-for-uec`
yusuke@mm-dhcp-124-238 llm-tutorial % 
```

---

実際に簡単なPythonコードを書いて、LLMを実行していこうと思います。少し古いですが、基本的なLLMとして `Llama-2-7b-chat-hf`を選択します。

```python
from transformers import pipeline
import torch

# 1. モデル名
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 2. モデルの作成。簡単のためにpipelineというツールを用いている。
# 2.1 torch_dtype=torch.float16は16bitで読み込み。後述。
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.float16)

# 3. プロンプト（入力文）：基本的にここを書き換えることがメインになる。
prompt = "Tell me a story about a dragon."

# 4. ここでプロンプトをLLMに入力して、出力を得る。
output = pipe(prompt, max_length=100, do_sample=False)
print(output[0]["generated_text"])
```

出力結果：
```
Tell me a story about a dragon.

Sure! Here is a story about a dragon:

Once upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as red as fire and wings that stretched as wide as the sky. He lived in a great, sprawling cave deep within a mountain, surrounded by piles of
```

ここでは、簡略化のために、細かい設定はほとんどデフォルトにしています。また、プロンプトを入力したら、テキストが返ってくるといった具合に、内部処理を`pipeline`を使用することで、一旦処理工程を隠しています。また、Transforemersは心配性なので、warningが出てくることがありますが、大半はお知らせのようなものなので、無視して大丈夫です。

以下、各行の説明です：

1. はじめに、モデル名を設定してください。私は、モデル名を変数で置いています。理由として、実験過程で複数のLLMを用いる場面が頻繁に発生するので、for文やコメントアウトでの切り替えなどを変数で指定することが多いからです。利用可能なモデル名は[こちら](https://huggingface.co/models?pipeline_tag=text-generation&sort=likes)で確認できます。タイトル部分（`〇〇/〇〇`になっている箇所）をコピペすれば利用可能です。

2. 次に、HuggingFace Transformersの`pipeline`を使用して、モデルをロードします。このとき、`torch_dtype=torch.float16`と記述するのは、LLMのサイズが少し大きいため、16bitのfloat型にキャストして読み込むことで、メモリ使用量を約半分にしています。float型が32bitの浮動小数点であることを思い出してください。

3. 次にプロンプトを記述していきます。おそらく皆さんが工夫できる箇所だと思います。LLMはこの記述内容に従って、出力を生成します。ここも基本的には変数に保存しておいたほうが、何回も試行可能です。

4. 最後に、プロンプトをモデルに入力することで、出力を得ます。

   - このとき、`output[0]["generated_text"]`のように出力が`List[Dict]`の形式になっていることに注意してください。これはHuggingFaceが全ての出力を辞書型へ返す用に設計されているためです。

   - リスト型で最初に`[0]`と指定するのは、後述するサンプリング型生成手法などで、複数文生成する場合にリストで生成結果を管理するためです。
   - `max_length=100`はプロンプト含めて、合計100トークン生成するという意味です。そのため、出力結果の最後が "surrounded by piles of..."で途切れています。生成終了条件は2つ、終了トークン`<eos>`等、あるいは指定されたトークンが出現するまでか、指定されたトークン数まで生成します。そのため、`'1+1=?`のようなプロンプトには`[{'generated_text': '1+1=?\n\nThe answer is 2.'}]`と短く答えてくれます。
   - `do_sample=False`は後述サンプリング型生成手法を使用するときに`True`にします。`False`にすることで、出力結果が一意に定まります。試しに何度か上記のコードを試してみてください。出力結果は固定です。

実際にモデル名やプロンプトを変更して、動作が変わることを確認してみてください。英語以外にも日本語だとどうなるのかの検証も面白いかもしれません。

> [!TIp]
>
> **`max_length`と`max_new_token`の違い：**
> 4番目の処理で、`max_length`を指定しました。これは<u>プロンプトを含めた</u>トークン数です。プロンプトが長い場合、例えばデータセットの内容を変数でプロンプトに代入したり、RAGのように外部から取得してきたデータをプロンプトに含めるとき、`max_length`が少ないと大半をプロンプトで消費してしまい、LLMが生成する箇所がほとんど残らなくなります。それを回避するため、<u>LLMが何トークン生成するのか</u>指定する`max_new_token`が用意されていますので、検証目的ならこちらを使用してもいいかもしれません。あるいは、`max_length`を長くするか。
> もう少し深堀りすると、LLMの出力が終了せず、永遠と文字を出力することがあります。さらに、長く生成すればするほど、メモリを消費するため、Out of Memoryなどハードウェア側のエラーが発生してしまいます。さらに、複数文一気に処理するとき、文数が揃っていたほうが処理がしやすいです。そのため、安定的にシステムを動かすために、プロンプトの長さを考慮したmax_lengthを指定することが多いようです。

**課題1：**モデルやプロンプトなどを変更して動作が変化することを確認せよ。目標は出力が変化することを確認してください。また、プロンプトのみ変更することで、目的の出力がされるか（端的に答えてくれるかなど）、ある程度制御できるか、試してみてください。

---

### 補足：もう少し詳細な動作の仕組み

先程は、`pipeline`を使うことで、処理工程を隠しました。実用上はこれで問題ないのですが、せっかくなので、もう少し深堀りして、どのような挙動になっているのか、確認してみます。

`pipeline`を使わない場合、コードは以下のようになります：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. モデル名
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 2. トークナイザとモデルのロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).eval()

# 3. プロンプト（入力文）
prompt = "Tell me a story about a dragon."

# 4. トークナイズ処理
inputs = tokenizer(prompt, return_tensors="pt")
## 実際には以下の処理を一気に行っている。
## token = tokenizer.tokenize(prompt)
## input_ids = tokenizer.convert_tokens_to_ids(tokens)
## input_ids_tensor = torch.tensor([input_ids])
## attention_mask = torch.ones_like(input_ids_tensor)
##
## inputs = {
##     "input_ids": input_ids_tensor,
##     "attention_mask": attention_mask,
## }

# 5. 生成
output_ids = model.generate(
  **inputs,
  max_length=100,
  do_sample=False,
)

# 6. トークン列をテキストに変換
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
## tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
## text = tokenizer.convert_tokens_to_string(tokens)
```

1番目と3番目はただの設定なので、重複していますが、2,4,5,6番目は少しややこしいです。順を追って確認します：

2. モデルに加えて、トークナイザをロードしています。トークナイザで入力文をID列に変換し、モデル内部で演算を行い、その結果を再び、トークナイザでID列から文字列へ変換します。


4. 次に、`tokenizer(prompt, return_tensors="pt")`でID列へ変換します。その結果以下のとおり文字列が`input_ids`の中に格納されている通り、ID列に変換されます。
   ```bash
   "Tell me a story about a dragon." -> 
   {'input_ids': tensor([[    1, 24948,   592,   263,  5828,  1048,   263,  8338,   265, 29889]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
   ```

   実際には、4のコード直下にコメントアウトしているとおり、いくつかの手順をまとめて行っています。

   ```python
   tokens = tokenizer.tokenize(prompt)
   # ['▁Tell', '▁me', '▁a', '▁story', '▁about', '▁a', '▁drag', 'on', '.']
   input_ids = tokenizer.convert_tokens_to_ids(tokens)
   # [24948, 592, 263, 5828, 1048, 263, 8338, 265, 29889]
   input_ids_tensor = torch.tensor([input_ids])
   # tensor([[24948,   592,   263,  5828,  1048,   263,  8338,   265, 29889]])
   attention_mask = torch.ones_like(input_ids_tensor)
   # tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
   ```

   はじめに、プロンプトは文なので、単語やそれよりも短いサブワードという単位に分割します。サブワードとは、統計的に頻出する部分文字列のことをいいます。今回の例では単語とサブワードが一致しているので、単語単位で分割されましたが、例えば長い単語`Supercalifragilisticexpialidocious`の場合、`['▁Super', 'cal', 'if', 'rag', 'il', 'ist', 'ice', 'xp', 'ial', 'ido', 'cious']`のように、いくつかのサブワードに分割されます。このように分割することで、LLM内部で扱う語彙サイズを削減できます。分割後、モデル側の処理工程ではトークン、言語側での処理工程ではサブワードと呼ぶのが一般的なようです。
   次に、各トークンをID列に変換します。ID列に変換することで、文をベクトルの計算へ落とし込みます。そのあと、pytorchの`tensor`型へ変換することで演算を手軽に行えるようにします。

   最後に`attention_mask`がありますが、これは一旦無視してもらって構いません。歴史的経緯で、複数文章を明示的に入力したいニーズが過去にありました。そのとき、文の区切れ目を認識する際に、1/0をフリップさせるように、`attention_mask`を設定していましたが、今回は複数文でも、内部的にはすべて1文とみなすため、すべて1になっています。

5. `model.generate`でLLMが計算を行います。その結果、

   ```python
   tensor([[    1, 24948,   592,   263,  5828,  1048,   263,  8338,   265, 29889,
               13,    13, 29903,   545, 29991,  2266,   338,   263,  5828,  1048,
              263,  8338,   265, 29901,    13,    13, 26222,  2501,   263,   931,
            29892,   297,   263,  2215, 29899,  2696,  2982, 29892,   727, 10600,
              263, 29154,   296,  8338,   265,  4257,  2522, 25350, 29889,  2522,
            25350,   471,   278, 10150,   322,  1556, 13988,  8338,   265,   297,
              599,   278,  2982, 29892,   411, 23431,   408,  2654,   408,  3974,
              322, 24745,   393, 16116,   287,   408,  9377,   408,   278, 14744,
            29889,   940, 10600,   297,   263,  2107, 29892, 26183,  1847, 24230,
             6483,  2629,   263, 14378, 29892, 22047,   491,   282,  5475,   310]])
   ```

   のようにID列が生成されます。今回は100トークンフルで生成されましたが、
   ```python
   tensor([[    1, 29871, 29896, 29974, 29896, 29922, 29973,    13,    13, 22550,
            29901, 29871, 29906,    13,    13,  1252,  9018,   362, 29901,    13,
            10401,   366,   788, 29871, 29896,   718, 29871, 29896, 29892,   278,
             1121,   338, 29871, 29906, 29889,     2]])
   # 1+1=?\n\nAnswer: 2\n\nExplanation:\n\nWhen you add 1 + 1, the result is 2.
   ```

   のように、短い場合もあります。ID 2というのが、終了トークンを今回の場合意味しているからです。その反対に両方とも先頭にID 1がついています。これは開始トークンを表しています。内部では開始位置と終了位置を明示します。

6. 最後に`tokenizer.decode`でID列から文へ変換します。`skip_special_tokens=True`は前述したように、モデル内部では開始位置と終了位置などのように特殊なトークンを使用するため、これを出力時に除去するかどうかの引数になっています。また、直下のコメントアウトのように`decode`処理も、まずサブワード列に変換してから文へ変換しています。
   ```python
   # 今回は下記のtensorが得られたとします。
   # tensor([[    1, 29871, 29896, 29974, 29896, 29922, 29973,    13,    13, 22550,
   #          29901, 29871, 29906,    13,    13,  1252,  9018,   362, 29901,    13,
   #          10401,   366,   788, 29871, 29896,   718, 29871, 29896, 29892,   278,
   #           1121,   338, 29871, 29906, 29889,     2]])
   tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
   # ['<s>', '▁', '1', '+', '1', '=', '?', '<0x0A>', '<0x0A>', 'Answer', ':', '▁', '2', '<0x0A>', '<0x0A>', 'Ex', 'plan', 'ation', ':', '<0x0A>', 'When', '▁you', '▁add', '▁', '1', '▁+', '▁', '1', ',', '▁the', '▁result', '▁is', '▁', '2', '.', '</s>']
   text = tokenizer.convert_tokens_to_string(tokens)
   # <s> 1+1=?\n\nAnswer: 2\n\nExplanation:\n\nWhen you add 1 + 1, the result is 2. </s>
   ```

   サブワード列を観察すると、先頭に`<s>`、末尾に`</s>`が付与されており、特殊トークンが付与されていることが確認できます。また、改行が`<0x0A>`で表現されていたり、Explanationという単語が`'Ex', 'plan', 'ation'`と3トークンに分割されています。また、`'▁you'`のように、一部の単語の冒頭に`▁`というアンダーバーのような記号が付与されています。大雑把に言えば、空白を表しています。そのため、`'Ex', 'plan', 'ation'`には`▁`が出現しませんが、これらサブワードを結合させれば`Explaination`という単語になり、`'When', '▁you', '▁add'`を結合させて、`▁`を取り除けば、`When you add`になるので、単語の区切れ目をうまく扱うことができます。他にも`@@`などで区切れ目や単語同士の結合箇所を表すモデルもあります。
   
   これらの処理は`''.join(tokens).replace('▁',' ')`のように簡単なPythonコードで書き換えも可能ですが、細かい処理やトークナイザの挙動の違いなどあるため、tokenizerの処理に任せたほうが楽です。

以上のように、`pipeline`でまとめた箇所を細かく読み解くと様々な処理が行われていますが、通常モデルを推論のみにしか使わないのであれば、このような細かい工程を自分で実装すると、コードが長くなったり例外処理などでバグが含まれてしまうかも知れませんので、`pipeline`を用いるほうがいいでしょう。原理だけ覚えてもらえたらいいです。

### 参考資料

- https://github.com/bitsandbytes-foundation/bitsandbytes
- https://zenn.dev/turing_motors/articles/2fd279f6bb25a4
- https://qiita.com/xxyc/items/fa6188b89fa9533dc674
- https://huggingface.co/docs/transformers/ja/main_classes/pipelines
- https://zenn.dev/hellorusk/articles/7fd588cae5b173
- https://qiita.com/lighlighlighlighlight/items/48ed6532c481d78a139d

---

## 1.1 出力を多様にする

先ほどまでのコードは何回実行しても一意な結果が得られました。これは`pipe(prompt, max_length=100, do_sample=False)`のように、`do_sample=False`としていたからです。試しに`do_sample=True`へ変更して何回か実行してみてください。

```python
from transformers import pipeline
import torch

# 1. モデル名
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 2. モデルの作成。簡単のためにpipelineというツールを用いている。
# 2.1 torch_dtype=torch.float16は16bitで読み込み。後述。
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.float16)

# 3. プロンプト（入力文）：基本的にここを書き換えることがメインになる。
prompt = "Tell me a story about a dragon."

# 4. ここでプロンプトをLLMに入力して、出力を得る。
output = pipe(prompt, max_length=100, do_sample=True) # 変更箇所
print(output[0]["generated_text"])
```

出力結果（1回目）：

```bash
Tell me a story about a dragon.

Dragons are mythical creatures that are often depicted as powerful, fire-breathing beings with wings and claws. They are often associated with magic, wisdom, and power. In this story, a young dragon named Ember lives in a hidden cave deep in the mountains. Ember is a curious and adventurous dragon, always eager to explore the world beyond his cave. One day, Ember
```

出力結果（2回目）：

```bash
Tell me a story about a dragon.

Sure! Here is a story about a dragon:

Once upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales that shone like gold in the sunlight and wings that stretched as wide as the horizon. He was a fierce and fearsome creature, with eyes that g
```

出力結果（3回目）：

```
Tell me a story about a dragon.

Once upon a time, in a land far, far away, there was a magnificent dragon named Scorch. Scorch was unlike any other dragon in the land. While most dragons were content to spend their days basking in the sun and hoarding treasure, Scorch was always on the move, exploring new lands and seeking out adventure.

One day, as Scorch was soaring through
```

このように、実行するたびに結果が変わりました。ところで`pipeline`等のTransformersのモジュールには、`num_return_sequences`という引数があります。何度も実行する代わりに、一括で出力を得る方法があります。

```python
...
# 4. ここでプロンプトをLLMに入力して、出力を得る。（注意：output-> outputsに変数名変更している）
outputs = pipe(prompt, max_length=100, do_sample=True, num_return_sequences=3) # 変更箇所
print(outputs)
# 以下出力結果
[
  {'generated_text': 'Tell me a story about a dragon.\n\nI want to hear a story about a dragon that is a little bit different from the usual ones. Can you tell me a story about a dragon who is not evil, but rather a protector of the land and its people?\n\nOf course! Here is a story about a dragon who is not evil, but rather a protector of the land and its people:\n\nOnce upon a time, in a far'}, 
  {'generated_text': 'Tell me a story about a dragon.\n\nOnce upon a time, in a far-off land, there was a magnificent dragon named Scorch. Scorch was unlike any dragon that had ever been seen before. He was covered in shimmering scales of gold and silver, and his wings were as wide as the horizon. He had a fiery mane that flowed like a river of flame, and his eyes glowed like embers from the'}, 
  {'generated_text': 'Tell me a story about a dragon.\n\nDragon Story\n\nOnce upon a time, in a land far, far away, there lived a magnificent dragon named Scorch. Scorch was unlike any other dragon in the land, for he had a heart of gold and a fierce determination to protect his home and the creatures that lived within it.\n\nScorch lived in a vast, sprawling cave system deep within a great mountain range.'}
]
```

このように、`do_sample=True, num_return_sequences=3`と、生成する文数を指定すると、多様な出力を同時に生成してくれます。

基本的にLLMの生成は、次のトークンとして最も確率の高いものを選択することにより、終了トークンが出現するまで生成し続けます。これをMAPデコーディング（Maximum a Posteriori decoding）あるいは、確定的デコーディング（Deterministic decoding）と呼びます。特に今回の場合、シンプルに一番確率の高い単語のみを選択しているのでGreedy decodingと呼びますが、用語が多くなって混乱するので、「こんな用語もあるんだ」くらいで十分です。重要なのは、`do_sample=False`のときの手法では、入力したプロンプトに対して出力が一意に定まるということです。そのため、`do_sample=False`には`num_return_sequences`という引数は使うことが出来ません。

一方`do_sample=True`の場合、生成結果が変わり、出力が一意に定まらなくなりました。これは確率的デコーディング（Stochastic decoding）といい、最も確率の高いトークンを選択するのではなく、若干のランダム性をもたせることによって、出力を多様にしています。実用上は、正直`do_sample=True`で十分ですが、いくつかのパラメータを操作することによって、「どの程度多様か」を調整できます。

以下のコードは`top_p=0.9, temperature=0.8`を4番目のコードに追加しています。この2つのパラメータを使って多様性を制御していきましょう。なお、目視で多様性を測るより、定量的に測るほうがわかりやすいので、ここではSelf-BLEU ([Zhu et al., 2018](https://dl.acm.org/doi/10.1145/3209978.3210080))を用いて、多様性を測ろうと思います。詳細は省きますがとりあえず、0から100で<u>スコアが低ければ低いほど</u>、各文が似ていないので、つまり多様であると考えてください。
`top_p`(0から1まで)と`temperature`（0より上の値をとれるが、1くらいまでで十分。思い切って2まで。）の値、必要に応じて`num_return_sequences`を変更して出力文の変化と、スコアがどのように変動するか試してみてください。（注意：あくまでランダムなため、数値は変動します。文数が多ければ多いほど安定しますが、今は傾向を掴むために、色々数値を変更して遊んでみてください。）

```python
from transformers import pipeline
import torch
import sacrebleu # 定量的に多様性を測るため、metricを追加

# 1. モデル名
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 2. モデルの作成。簡単のためにpipelineというツールを用いている。
# 2.1 torch_dtype=torch.float16は16bitで読み込み。後述。
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.float16)

# 3. プロンプト（入力文）：基本的にここを書き換えることがメインになる。
prompt = "Tell me a story about a dragon."

# 4. ここでプロンプトをLLMに入力して、出力を得る。
outputs = pipe(prompt, max_length=100, do_sample=True, num_return_sequences=3,
              top_p=0.9, temperature=0.8) # 追記箇所
print(outputs)

# (Option) Self-BLEUの計算。低ければ低いほうがいい。
scores = []
generated_texts = [output["generated_text"].strip() for output in outputs]
for i, hypo in enumerate(generated_texts):
    refs = generated_texts[:i] + generated_texts[i+1:] # 他の全てを参照とする
    bleu = sacrebleu.corpus_bleu([hypo], [refs]) # sacrebleuはリストを受け取る
    scores.append(bleu.score)

# Self-BLEUの結果。低ければ低いほうが、多様性が高いので注意。
for i, score in enumerate(scores):
    print(f"Self-BLEU for output {i+1}: {score:.2f}")
print(f"Average Self-BLEU: {sum(scores)/len(scores):.2f}")
```

出力結果（top_p=0.9, temperature=0.8の場合）：

```python
[
  {'generated_text': 'Tell me a story about a dragon.\n\n---\n\nOnce upon a time, in a land far, far away, there lived a magnificent dragon named Scorch. Scorch was unlike any dragon that had ever been seen before. He was covered in shimmering scales that glistened in the sunlight, and his wings stretched out wide enough to block out the entire sky.\n\nScorch lived in a great, sprawling cave deep within'},
  {'generated_text': 'Tell me a story about a dragon.\n\n"The dragon\'s name was Scorch, and he was the most fearsome creature in all the land. He had scales as black as coal, wings as wide as a castle, and a fiery breath that could melt steel. Scorch lived in a great cave deep within a mountain, and he spent his days sleeping and breathing fire.\n\nOne day, a brave knight named Sir Edward decided to'},
  {'generated_text': "Tell me a story about a dragon.\nI'd be happy to! Here is a story about a dragon:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as black as night and wings that shone like the brightest stars. He was a fierce creature, feared by all who lived within his domain"}
]
Self-BLEU for output 1: 19.34
Self-BLEU for output 2: 19.36
Self-BLEU for output 3: 31.67
Average Self-BLEU: 23.46
```

出力結果（top_p=0.1, temperature=0.8の場合）：

```python
[
  {'generated_text': 'Tell me a story about a dragon.\n\nSure! Here is a story about a dragon:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as red as fire and wings that stretched as wide as the sky. He lived in a great, sprawling cave deep within a mountain, surrounded by piles of'}, 
  {'generated_text': 'Tell me a story about a dragon.\n\nSure! Here is a story about a dragon:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as red as fire and wings that stretched as wide as the sky. He lived in a great, sprawling cave deep within a mountain, surrounded by piles of'}, 
  {'generated_text': 'Tell me a story about a dragon.\n\nSure! Here is a story about a dragon:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as red as fire and wings that stretched as wide as the sky. He lived in a great, sprawling cave deep within a mountain, surrounded by piles of'}
]
Self-BLEU for output 1: 100.00
Self-BLEU for output 2: 100.00
Self-BLEU for output 3: 100.00
Average Self-BLEU: 100.00
```

出力結果（top_p=0.5, temperature=0.8の場合）：

```python
[
  {'generated_text': 'Tell me a story about a dragon.\n\nDragon Story\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was unlike any other dragon in the land, for he had a heart of gold and a fierce determination to protect his home and the creatures within it.\n\nScorch lived in a vast, sprawling cave system deep within a mountain range. The cave was filled'},
  {'generated_text': 'Tell me a story about a dragon.\n\nSure! Here is a story about a dragon:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as black as coal and eyes that glowed like embers. He lived in a great, sprawling cave system deep within a mountain range, surrounded by piles'}, 
  {'generated_text': 'Tell me a story about a dragon.\n\nSure! Here is a story about a dragon:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as red as fire and wings that stretched wider than the horizon. He lived in a great cave high up in the mountains, surrounded by a hoard of treasure'}
]
Self-BLEU for output 1: 46.92
Self-BLEU for output 2: 46.91
Self-BLEU for output 3: 37.63
Average Self-BLEU: 43.82
```

出力結果（top_p=0.5, temperature=0.1の場合）：

```python
[
  {'generated_text': 'Tell me a story about a dragon.\n\nDragon stories are a classic tale that has been told and retold for centuries. They are filled with magic, adventure, and of course, dragons. Here is a story about a dragon that I hope you will enjoy:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was unlike any dragon that had ever been seen before. He'},
  {'generated_text': 'Tell me a story about a dragon.\n\nSure! Here is a story about a dragon:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as red as fire and wings that stretched as wide as the sky. He lived in a great, sprawling cave deep within a mountain, surrounded by piles of'}, 
  {'generated_text': 'Tell me a story about a dragon.\n\nDragon stories are a classic tale that has been told and retold for centuries. They are filled with magic, adventure, and of course, dragons. Here is a story about a dragon that I hope you will enjoy:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was unlike any dragon that had ever been seen before. He'}
]
Self-BLEU for output 1: 42.13
Self-BLEU for output 2: 42.13
Self-BLEU for output 3: 100.00
Average Self-BLEU: 61.42
```

出力結果（top_p=0.5, temperature=1.5の場合）：

```python
[
  {'generated_text': "Tell me a story about a dragon. I want to be amazed.\nA few days ago, a man named Tom was wandering through a dense forest when he stumbled upon a magnificent dragon. The dragon was unlike any Tom had ever seen before, with scales that shimmered in the sunlight like diamonds and wings that stretched as wide as the trees themselves.\nTom was amazed by the dragon's size and beauty, but he was"},
  {'generated_text': "Tell me a story about a dragon. 🐉\n\nSure, here's a story about a dragon:\n\nOnce upon a time, in a far-off land, there lived a magnificent dragon named Scorch. Scorch was the largest and most powerful dragon in all the land, with scales as black as coal and eyes that glowed like embers. He lived in a great cave deep within a towering mountain, surrounded"},
  {'generated_text': 'Tell me a story about a dragon.\nAsked by: Sophie M\nOnce upon a time, in a far-off land, there was a magnificent dragon named Scorch. Scorch was unlike any dragon that had ever been seen before, for he was covered in shimmering scales of gold and silver, and his wings were as wide as the sun.\n\nScorch lived in a great, sprawling cave system deep within a towering'}
]
Self-BLEU for output 1: 14.11
Self-BLEU for output 2: 14.09
Self-BLEU for output 3: 19.06
Average Self-BLEU: 15.75
```

出力結果（top_p=1.0, temperature=10の場合）：

```python
[
  {'generated_text': 'Tell me a story about a dragon. This... dragone doesn�� like its home because itв�eton; a cage in his chom where his homeвъсреа where, like other creature with out feet the and wasвore but. вThis must the a small dank is. All is in pitch yoke y� dragonly see th dragО him through window bars hd th, dark gвt that has t in their p of c he'},
  {'generated_text': 'Tell me a story about a dragon. It had a... 5 Anunciados (Dragnelli # Drag) are dragoni? (Answerer... As much happiness and comfort asto my mother... Tame, sweet-facedyouth... Layw from Manaos, his mother having tchenlse. As Much happ... He t... This story doesnï¼‘ Dragon born to an old wicked wo... 5Anunciao da Morrã'}, 
  {'generated_text': 'Tell me a story about a dragon. This should not feel forced?’\nYou say ‘Ok Sure. How about the story on dragoni?” Dragone smiled broad the smile'}
]
Self-BLEU for output 1: 11.35
Self-BLEU for output 2: 11.36
Self-BLEU for output 3: 6.67
Average Self-BLEU: 9.80
```

なんとなく、感覚つかめましたか？

top_pもtemperatureも低いほうが多様性が減少し、top_p=0.1, temperature=0.8の場合では、全てが同じ出力になりました。一方、高ければよいというわけでもなく、top_p=1.0, temperature=10の場合では出力が崩壊しています。このように、多様にすればするほど、自然な文から離れる場合があるので、パラメータ調整が肝になってきます。基本的に、top_p=0.9, temperature=0.8とかであれば、適度に多様でいい感じの文が生成される傾向があるようですが、あくまで傾向なので、モデルやタスクによって話が変わります。私が研究で使う場合、再現性がある程度必要な場面が多いので、最初は低いパラメータからどんどん大きくしていきます。両方同時に変更することはせずに、ガスバーナーのガスと空気のように、片方ずつ少しずつ調整して感覚を掴んでください。

> [!Tip]
>
> 乱数を使っているため、出力結果が毎回変動してしまいます。そのため、乱数を固定して、出力の再現性を担保する場合が研究などではあります。乱数の固定方法として楽なのは、Transformersの`set_seed()`関数を使うことです。
>
> ```python
> from transformers import pipeline, set_seed
> 
> set_seed(0)  # 適当なSEED値を入力して固定
> ...
> ```



---

### 補足：もう少し詳細な動作の仕組み

先程は`top_p`と`temperature`を<u>出力を多様にするパラメータ</u>とぼかして説明しましたが、もう少しだけ詳しく説明します。そもそもLLMは「次のトークンを予測する」を繰り返すことで文を生成します。各生成時には、各語彙の全てに対する確率を計算し、その中でもっとも良いものが選ばれるのがMAP decodingでした。もしランダム性をもたせるのであれば、この単語選択時に、最も確率が高いものを選ぶのではなく、全ての語彙から適当なトークンを選択すればよいです。しかしそれでは文が崩壊してしまいますので、せっかくなら、この確率を使いたいです。次のような確率を考えてみましょう。

```python
# Tell me a story about a dragon.が与えられたときの、次の単語の確率と想定
[
    ("the", 0.25),
    ("a", 0.20),
    ("dragon", 0.15),
    ("flew", 0.10),
    ("over", 0.08),
    ("and", 0.07),
    ("castle", 0.06),
    ("into", 0.04),
    ("flames", 0.03),
    ("darkness", 0.02),
]
```

確率なので、合計1になっていることを確認してください。モックで10個のみですが、実際には数万単位であります。普通なら一番確率が高い"the"を選択しますが、`top_p`の場合、累積確率（上から順に確率を足した合計）の中からランダムに単語を選択します。もし`top_p=0.8`なら、"the"から"and"まで足した0.85（"over"までなら0.78なので）までの単語の中からランダムに生成するということを繰り返します。これをTop-p (nucleus) sampling ([Holtzman et al., 2020](https://openreview.net/forum?id=rygGQyrFvH))といいます。

> [!Note]
>
> Top-p samplingのみを今回は対象にしますが、他にも色々なSampling decoding方法があります。例えば<u>上位k件までを対象</u>とするTop-k sampling ([Fan et al., 2018](https://aclanthology.org/P18-1082/))は、top_k=3などと指定すれば、"dragon"までの3つの単語の中からランダムに選択します。
>
> ```python
> pipe(prompt, max_length=100, do_sample=True, num_return_sequences=3,
>               top_p=1.0, top_k=3, temperature=1.0)
> ```
>
> のように`top_k`を追加すれば、自動的にTop-k Samplingになります。ここでのポイントは`top_p`などのパラメータと組み合わせて使用可能なところです。累積確率で`top_p`のしきい値までトークンを選択するか、`top_k`のしきい値に達するまでのどちらかまで単語を選択するなどの複雑な指定ができます。概念を考えれば、なんだかできそうでしょう？利点として、`top_p`の場合、どの単語の選択でもいい場合、全ての単語の確率がほぼ似たような分布になります。その場合、候補の単語が膨大になり、明らかにふさわしくない変な単語が選択されるかもしれません。それを回避するために、`top_k`でマックスで扱う単語数を決めておくと、このような確率がほぼ均等な場合に対応可能となります。
>
> `top_p`も`top_k`も確率分布からしきい値を決めて、その範囲の単語を選択していましたが、全ての語彙からランダムに単語を選ぶ方法をAncestral Sampling ([Robert, 1999](https://link.springer.com/book/10.1007/978-1-4757-4145-2))といいます。以下のように設定すれば動きます。
>
> ```python
> pipe(prompt, max_length=100, do_sample=True, num_return_sequences=3,
>               top_p=1.0, top_k=0, temperature=1.0) # top_k=0は設定しないという意味。つまり全ての単語を対象とする
> ```
>
> 他にいい方法は考えられないでしょうか？先ほど`top_p`と`top_k`を併用するモチベーションと利点について、 
>
> > `top_p`の場合、どの単語の選択でもいい場合、全ての単語の確率がほぼ似たような分布になります。その場合、候補の単語が膨大になり、明らかにふさわしくない変な単語が選択されるかもしれません。
>
> と説明しました。組み合わせる以外にも、例えば、確率がしきい値以下の単語を除去することで、明らかに不適な単語を除去する方法が考えられます。このようにしきい値で下限を決める方法をEpsilon Sampling ([Hewitt et al., 2022](https://aclanthology.org/2022.findings-emnlp.249/))いいます。
>
> ```python
> pipe(prompt, max_length=100, do_sample=True, num_return_sequences=3,
>               epsilon_cutoff=0.05, temperature=1.0) # epsilon_cutoffで下限を指定。0.02がよく用いられている。
> ```
>
> この場合、"into", "flames", "darkness"の単語の確率が0.05未満なので、これらを除外した中からランダムに選択されます。
>
> 他になにかいいアイデアはありますか？もし思いついて実行してベンチマークで良い値が取れたら論文を書けるので、時間があったら考えてみてください。他に細かい設定などは[公式ドキュメント](https://huggingface.co/docs/transformers/ja/main_classes/text_generation)にとても詳細に書かれているので、これも時間があれば確認してみたら、面白い気づきや発見があるかもしれません。

> [!Warning]
>
> ここでは、簡略化するためランダムと言及していますが、実際には選択した単語を元に再び確率分布を振り直しています。Top-p Sampling (p=0.8)の場合、
>
> ```python
> # 元の分布
> [
>     ("the", 0.25),
>     ("a", 0.20),
>     ("dragon", 0.15),
>     ("flew", 0.10),
>     ("over", 0.08),
>     ("and", 0.07),
>     ("castle", 0.06),
>     ("into", 0.04),
>     ("flames", 0.03),
>     ("darkness", 0.02),
> ]
> # Top-p Sampling (p=0.8)で確率分布を振り直した後
> [
>     ("the", 0.294),
>     ("a", 0.235),
>     ("dragon", 0.176),
>     ("flew", 0.118),
>     ("over", 0.094),
>     ("and", 0.082),
> ]
> # どちらも総和は1になる
> ```
>
> となり、ここから確率分布に基づいたサンプリング（[Multinomial sampling](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html)）を行います。つまり、"the"が出る確率は0.294, "flew"が出る確率は0.118といった具合です。
>
> ただしランダムという単語の使い方は結構曖昧です。例えばTop-kの原著論文 ([Fan et al., 2018](https://aclanthology.org/P18-1082/)) にはrandomとだけしか書かれていなかったりしますし、実装によっては異なる結果になるかもしれないので、実装を直接見てみると面白いかもしれません。
>
> さらに今回beamの説明も省いたので、若干曖昧な説明のままになっています。端的にbeamを説明すると、文生成において、「次のトークンを予測する」を繰り返すことで、確率の掛け算になります。このとき、その時に一番良いものを選択し続けたら、もしかしたら数手先で頓死/ドボンとなるような単語の組み合わせが発生するかもしれません。将棋やオセロを思い出してください。直前で自分が有利になるような手ばかり指していると、行き詰まったり不利になることがありますよね。このように数手先を読んで行動するように、Decodingでも数手先を読んで生成したほうが良い文が完成することがあります。ざっくり言うと、この数手先を読むことをbeamと表現します。通常は`num_beam=5`のように設定しますが、Samplingの場合`num_beam=1`と明示的に設定するのが正しいですが、実用上そのような厳密性は蛇足だし、なにかが変わるというわけでもないので、正直、どちらでもいいです。



次に`temperature`について説明します。今までは確率分布が同じでしたが、もし、"the"が0.8などいきなり大きな確率（我々はピーキーと呼んでいる）になっている場合、せっかくTop-p Sampling (p=0.8)でも1つしか単語が選択されません。そのため、<u>もう少しなだらかな分布にする</u>ことで、Top-pなんだけど単語の選択肢を増やすことができそうです。また、Sampling Decodingは実際には、*確率分布に基づいたサンプリング*であるため、*なだらかな分布*にすることで、下位の単語が選ばれやすくなるような工夫が行われています。



ここから少し物理の話をします。めんどくさい方は読み飛ばしてもらって構いません。

高校物理で習った[シャルルの法則](https://w3e.kanazawa-it.ac.jp/math/physics/high-school_index/thermodynamics/theory_of_gases/henkan-tex.cgi?target=/math/physics/high-school_index/thermodynamics/theory_of_gases/Charles-s_law.html)を覚えていますか？

```math
\frac{体積（V）}{温度（T）} = 一定
```

シャルルの法則とボイルの法則を組み合わせることで、<u>理想気体における状態方程式</u>が成り立ちました。忘れた人や習ってない人は「そうなんだ」で、読み流してください。直感的に考えれば、温度が高い方が膨張していきそうというのはなんとなく分かると思います。これは気体を熱することで、温度が高くなると気体中の分子がだんだん活発に運動を始めるためです。寒いとアイスは個体ですが、溶けたら液体、やがて蒸発していく様子を想像していただければ、運動しているんだなってことはわかると思います。また、もう一つ直感的なこととして、水も100度になったらいきなり蒸発して、すべてなくなるわけではないように、徐々に蒸発していきます。これは各分子がどれくらい運動するか一定ではない、つまり個体差があるため、運動量の高いものから蒸発していきます。しかし、常温の水からも蒸発するように、一定ではなくこれらの例から（１）温度の変化はすべての気体分子に等しく影響を与える。（２）各分子には個体差がある。という2つのことが直感的に考えられます。このような関係は分布問題として捉えることができ、これらをモデリングしたものが[ボルツマン分布](https://ja.wikipedia.org/wiki/%E3%83%9C%E3%83%AB%E3%83%84%E3%83%9E%E3%83%B3%E5%88%86%E5%B8%83)です。要点だけ抑えた簡単な式を書きます。[Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%9C%E3%83%AB%E3%83%84%E3%83%9E%E3%83%B3%E5%88%86%E5%B8%83)から引っ張ってきています。

```math
p_i = \frac{exp(-q_i/kT)}{\sum^M_{j=1} exp(-q_j/kT)}
```

ややこしいでしょ？ここで大事なのは、すべての各確率qに対し、温度を考慮した新しい確率pへと変換してるところです。つまり、温度に変化があったら、すべての分子に対して同等の変化が加わるし、それらの合計量は1で、総和には変化がないということをうまくモデリングできています。この説明を厳密にボルツマン分布の研究をしている人がみたら、嘘が混ざっていると言われるかもしれませんが、あくまで直感的な理解としての説明なので、これくらいの理解度で十分です。

さて、LLMの`temperature`の話に戻ります。実用上はとても簡単です。それぞれの単語の確率をTで割って（1/Tを掛けて）あげます。それらの確率を元に、再び確率分布を振り直すことで*なだらかな分布*を得ることができます。

```math
p_i = \frac{exp(q_i/T)}{\sum^{|V|}_{j=1} exp(q_j/T)}
```

ここで、|V|は語彙数です。同様に、元の各トークンの確率qを温度で割り、確率分布を振り直すことで、新しい確率pが得られます。ほぼ式はボルツマン分布と一緒ですね。ボルツマン分布のkは定数なので、k=-1としたら、等価になると思います。コードで実装すると以下のようになります。

```python
import math

q = [("the", 0.25), ("a", 0.20), ("dragon", 0.15), ("flew", 0.10), ("over", 0.08), 
     ("and", 0.07), ("castle", 0.06), ("into", 0.04), ("flames", 0.03), ("darkness", 0.02)]
T = 1.5 # 一旦1.5に設定。

exp_q = [0]*len(q) # 一旦分子を全部計算する。
p = [None]*len(q) # pを格納するリストをあらかじめ作っておく

# ここで分子を一気に計算
for i in range(len(q)):
  exp_q[i] = math.exp(math.log(q[i][1]) / T) # [1]なのはタプルだから、確率のみ取得したい

# exp_qをsumで割って、pを求める。
for i in range(len(q)):
  p[i] = (q[i][0], exp_q[i] /sum(exp_q))
print(p)

# [('the', 0.1952), ('a', 0.1683), ('dragon', 0.1389), ('flew', 0.1060), ('over', 0.0913),
#  ('and', 0.0836), ('castle', 0.0754), ('into', 0.0575), ('flames', 0.0475), ('darkness', 0.0363)]

```

少しややこしいですが、`math.exp(math.log(q[i][1]) / T)`で一度logを取ってから、再びexpをしています。これはもともとの確率分布がすでにexpの形になっているため、logを取ることで元のスコアに戻してから、再度expをかけています。

> [!warning]
>
> ややこしいポイントとして、まず、T=1のとき、`temperature`を意識しない通常の確率の割当になります。このような一連の処理は[`softmax`](https://ja.wikipedia.org/wiki/%E3%82%BD%E3%83%95%E3%83%88%E3%83%9E%E3%83%83%E3%82%AF%E3%82%B9%E9%96%A2%E6%95%B0)関数という名前がついています。`temperature`の操作では、結局のところsoftmax関数の派生形である[`温度付きsoftmax`](https://qiita.com/nkriskeeic/items/db3b4b5e835e63a7f243)関数の適用を目指しています。重要なこととして、各要素をn倍してもsoftmax後の値には変化がありません。



もう少し別な言葉で言うと、





### 参考文献

- https://github.com/tehhuu/Self-BLEU/blob/master/Self-BLEU.py
  - めんどくさかったので、若干ChatGPTに修正してもらった。おおまかな挙動は変わらないはず。
- https://cyberagent.ai/blog/research/16115/
- https://note.com/npaka/n/n5d296d8ae26d
- https://techblog.a-tm.co.jp/entry/2023/04/24/181232
- https://qiita.com/suzuki_sh/items/8e449d231bb2f09a510c
- https://zenn.dev/hellorusk/articles/1c0bef15057b1d
- https://huggingface.co/docs/transformers/main_classes/model#transformers.generation_utils.GenerationMixin.generate
- 確率の適当な例をめんどくさかったので、ChatGPT（GPT-4o）に作ってもらった。
  - プロンプト：「top_pとtop_kの話をしたいので、10個程度でトークンの確率っぽいリストを作ってください」

- https://arxiv.org/abs/2410.15021
- https://huggingface.co/docs/transformers/v4.28.1/generation_strategies
- https://github.com/naist-nlp/mbrs/tree/main
- https://qiita.com/suzuki_sh/items/8e449d231bb2f09a510c
- https://d-engineer.com/netsuriki/boiru.html

ボランティア募集：言葉で説明はしんどいので、誰か図をつけてくれたら嬉しいです...







はじめに、補足で内部の挙動についてのレクチャー

次にquantization

そのあとに、



入力を複数受けるためです。

```python
prompts = ["Tell me a story about a dragon.",
          "Tell me a story about a phantom."]
output = pipe(prompts, max_length=100, do_sample=False)
print(output[0]["generated_text"])
print(output[1]["generated_text"])
```





export HF_HOME="/cl/home2/share/huggingface"

export HF_HUB_CACHE="/cl/home2/share/huggingface/hub"

export HF_ASSETS_CACHE="/cl/home2/share/huggingface/assets"















