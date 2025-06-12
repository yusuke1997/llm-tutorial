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

---

## 1.1 出力を多様にする

先ほどまでのコードは何回実行しても一意な結果が得られました。これは



あと

```bash
uv pip install vllm
uv pip install streamlit
```



はじめに、補足で内部の挙動についてのレクチャー

次にquantization

do_sampleについてかな



入力を複数受けるためです。

```python
prompts = ["Tell me a story about a dragon.",
          "Tell me a story about a phantom."]
output = pipe(prompts, max_length=100, do_sample=False)
print(output[0]["generated_text"])
print(output[1]["generated_text"])
```





内部での処理について。















export HF_HOME="/cl/home2/share/huggingface"

export HF_HUB_CACHE="/cl/home2/share/huggingface/hub"

export HF_ASSETS_CACHE="/cl/home2/share/huggingface/assets"















