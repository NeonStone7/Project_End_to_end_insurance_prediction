import pandas as pd
import boto3
from io import BytesIO, StringIO

def s3_retrieval(s3_client, bucket, file_name):

    # get object
    s3_clientobj = s3_client.get_object(Bucket=bucket, Key=file_name)
    print('Retrieved object')

    # Get the StreamingBody object from the response
    body = s3_clientobj['Body']

    # Read the bytes data from the StreamingBody
    csv_bytes = body.read()
    print('Read bytes')

    # Convert the bytes data to a pandas DataFrame
    csv_buffer = BytesIO(csv_bytes)

    df = pd.read_csv(csv_buffer)

    # Now df contains your CSV data as a DataFrame
    print('Success!')

    return df

def write_to_s3(df, s3_client, bucket_name, key_name):

    with StringIO() as csv_buffer:
    
        df.to_csv(csv_buffer, index = False)
        
        response = s3_client.put_object(Bucket = bucket_name,
                                    Key = key_name,
                                    Body = csv_buffer.getvalue())
        
    
    status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    
    if status == 200:
        
        print('Success!')
        
    else:
        
        print('Failed')