from ml.train import train_nst

async def run_nst(content_file, style_file):
    content_path = f"data/content/{content_file.filename}"
    style_path = f"data/style/{style_file.filename}"

    with open(content_path, "wb") as f:
        f.write(await content_file.read())

    with open(style_path, "wb") as f:
        f.write(await style_file.read())

    output_path = "outputs/result/output.png"
    train_nst(content_path, style_path, output_path)

    return output_path
