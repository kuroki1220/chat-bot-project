import os
import argparse
import logging
from process_qa import process_and_store_qa_data

# ロガー設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB index from QA CSV")
    parser.add_argument("--csv", type=str, default="qa_data.csv", help="読み込むCSVファイル名 (data_processing配下)")
    parser.add_argument("--collection", type=str, default="internal_qa_collection_v2", help="ChromaDBコレクション名")
    args = parser.parse_args()

    # data_processingディレクトリの絶対パスを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, args.csv)

    if not os.path.exists(csv_path):
        logger.error(f"指定したCSVファイルが見つかりません: {csv_path}")
        return

    logger.info(f"=== ChromaDB インデックス再構築を開始します ===")
    logger.info(f"使用CSV: {csv_path}")
    logger.info(f"コレクション名: {args.collection}")

    try:
        process_and_store_qa_data(csv_path, collection_name=args.collection)
        logger.info("✅ インデックス再構築が完了しました。")
    except Exception as e:
        logger.exception(f"❌ インデックス構築中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
