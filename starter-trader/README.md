# Starter Trader
https://blog.shidokamo.com/starter-trader

## インストール
pipenvでパッケージを管理しています。以下のようにパッケージをインストールできます。
pipenvを使わない場合は、Pipfileに記載されているパッケージを何らかの手段でインストールしてください。

```
make install
```

## 設定
.env というファイルを用意して、以下のような設定をかいてください。

```
# Google Cloud Functions v2 にインストールする場合以下の設定が必要
NAME=xxxxxxxxx
REGION=us-central1
PROJECT=xxxxxxxxx
SCHEDULER_SERVICE_ACCOUNT=xxxxxxxxxxxxxxx@${PROJECT}.iam.gserviceaccount.com

# 設定
QUOTE=USDC
 
COMMENT="COMMET for Google Cloud Functions v2 deployment"
BASE=ZEC
SPLIT_ORDERS=400
LEVERAGE=1.5
FREQ=1m
QUANTILE=0.9
WINDOW_HOURS=0.333
DIP=0.95
TAKE_PROFIT=1.025
SELL_ORDER_SPREAD=0.001
BUY_ORDER_SPREAD=0.002
POSITION_TIMEOUT_HOURS=4
BUY_PARTIAL_FILLS_TIMEOUT_HOURS=0.1
```

## API key
localpackage ディレクトリの中に、config.json というファイル名で、APIキーを入れてください。
これは、Hyperliquid公式の SDK で使われているものと同じです。

https://github.com/hyperliquid-dex/hyperliquid-python-sdk?tab=readme-ov-file#configuration

## ログ
PRODという環境変数が設定されている場合、JSON linesでログを吐きます（GCPに最適化されています）

## ローカルでの実行
pipenvをインストールしている場合は以下のコマンドでローカルで実行できます。

```
make run
```

## デプロイ
```
make deploy
```

## 定期実行の設定
GCP Cloud Scheduler で定期実行を設定できます。

```
make job
```

## 注意
このコードはスポット市場には対応していないので注意してください。
