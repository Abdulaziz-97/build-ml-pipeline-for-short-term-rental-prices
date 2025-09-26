#!/usr/bin/env python
"""
performs basic cleaning on the data and save the results in Weights & biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    logger.info(f"Fetching artifact {args.input_artifact}")
    artifact_local_path= run.use_artifact(args.input_artifact).file()

    df= pd.read_csv(artifact_local_path)


    logger.info(f"Dataset shape: {df.shape}")


    logger.info("Dropping outliers")
    
    idx= df['price'].between(args.min_price, args.max_price)
    df= df[idx].copy()

    logger.info("converting last_review to datetime")
    df['last_review']= pd.to_datetime(df['last_review'])
    logger.info("Dataset shape after converting last_review to datetime: {df.shape}")

    logger.info("filtering by geolocation - removing data points outside NYC area")
    # More precise NYC boundaries: Manhattan, Brooklyn, Queens, Bronx, Staten Island
    nyc_mask = (
        (df['latitude'] >= 40.4774) & (df['latitude'] <= 40.9176) & 
        (df['longitude'] >= -74.2591) & (df['longitude'] <= -73.7004)
    )
    outliers_count = len(df[~nyc_mask])
    logger.info(f"Found {outliers_count} data points outside NYC area")
    df = df[nyc_mask].copy()
    logger.info(f"Dataset shape after NYC area filtering: {df.shape}")
    
    output_file= "clean_sample.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved cleaned dataset to {output_file}")

    logger.info("Uploading artifact to Weights & Biases")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type= str,
        help= "The input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type= str,
        help= "The output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type= str,
        help= "The output type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type= str,
        help= "The output description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type= float,
        help= "The minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type= float,
        help= "The maximum price",
        required=True
    )


    args = parser.parse_args()

    go(args)
