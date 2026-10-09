def chunk_text(text:str,chunk_size:int=500,overlap:int=50):

    """
    words=text.split()
    chunks=[]
    start=0
    chunk_id=1
    while start<len(words):
        end=start+chunk_size
        chunk=" ".join(words[start:end])
        chunks.append({"chunk_id":chunk_id,"text":chunk})
        chunk_id+=1
        start=end-overlap
    return chunks"""

    lines=text.splitlines()
    chunks=[]
    current_section=None
    current_text=[]
    chunk_id=1

    for line in lines:
        line=line.strip()
        if not line:
            continue
        if (len(line.split())<=5 and not line.endswith(".")):
            if current_section and current_text:
                chunks.append({
                    "chunk_id":chunk_id,
                    "section":current_section,
                    "text":" ".join(current_text)
                })
                chunk_id+=1
            current_section=line
            current_text=[]
        else:
            current_text.append(line)

    if current_section and current_text:
        chunks.append({
            "chunk_id":chunk_id,
            "section":current_section,
            "text":" ".join(current_text)
        })

    return chunks