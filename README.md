# LLMチュートリアル@UEC

LLMの使い方に関する講座です。

---

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

### 参考資料
- https://docs.astral.sh/uv/
- https://zenn.dev/watanko/articles/8400df75e4aab5
- https://dev.classmethod.jp/articles/uv-unified-python-packaging-explained/
- https://qiita.com/yuya-0405/items/41c8153530d5542bee16

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

---

## 1. Transformersの使い方

- https://github.com/bitsandbytes-foundation/bitsandbytes
- https://zenn.dev/turing_motors/articles/2fd279f6bb25a4
- https://qiita.com/xxyc/items/fa6188b89fa9533dc674
- 



ローカルLLMを動かしていこうと思います。 [HuggingFace](https://huggingface.co/models) で大半のLLMが公開されているので、はじめに正式にサポートしている[Transformers](https://github.com/huggingface/transformers) ([Wolf et al., 2020](https://aclanthology.org/2020.emnlp-demos.6/)) を使用します。

> [!NOTE]
>
> Transformersを使用することが一般的ですが、通常のコンピュータで実行するには2つの課題があります。速度とメモリです。そのため計算方法を効率化した[vLLMs](https://github.com/vllm-project/vllm) ([Kwon et al., 2023](https://dl.acm.org/doi/10.1145/3600006.3613165)) 等を使うことで、推論速度を高速化したり、BitsAndBytes ([Dettmers et al., 2022](https://openreview.net/forum?id=dXiGWqBoxaD)) 等の量子化技術でメモリ削減する手法が開発されています。これらについては、後々詳細な説明を行います。

はじめに、下記コマンドでtransformersをinstallしてください。注意点として、`uv` を環境構築に使用しているため、`uv pip` のように`uv` を先頭に記述してください。`uv` を使用すると、installが高速になります。

```bash
uv pip install transformers
uv pip install "transformers[ja]"
uv pip install "transformers[sentencepiece]"
```

これで、Transformersのinstallは完了です。

---

### 補足: HuggingFaceのアカウント作成

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

実際に簡単なPythonコードを書いて、LLMを実行していこうと思います。































