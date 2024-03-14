from hashlib import sha256
import re
from urllib.parse import urlparse, parse_qsl, unquote_plus
import pandas as pd
from unicodedata import normalize
import pickle
import json

class Document:

    def __init__(
        self, 
        text, 
        doc_id=None, 
        url=None, 
        timestamp=None, 
        source_type=None,
        translation_type="sentence",
        sent_tokenization_model=None,
        **kwargs,
    ):
        self.text = normalize('NFKC', text).lower()
        self.url = url
        self.timestamp = timestamp
        self.source_type = source_type
        self.templated_text = self.text
        self.translated_text = self.text
        self.translation_complete = False
        self.source = self.get_source(self.url)
        self.sent_tokenization_model = sent_tokenization_model

        if not doc_id:
            self.doc_id = self.set_doc_id()
        else:
            self.doc_id = doc_id

        self.translation_type = translation_type

        if translation_type == "chunk":
            self.chunks = self.split_into_chunks()
        elif translation_type == "sentence":
            self.sentences = self.split_into_sentences()
        else:
            raise Exception("`translation_type` not supported. 2 types supported: `chunk` & `sentence`")

    def set_doc_id(self):
        string_to_hash = str(self.url) + str(self.timestamp) + self.text + str(self.source_type)
        return sha256(string_to_hash.encode('utf-8')).hexdigest()

    @staticmethod
    def normalize_url(url):
        if not url.startswith("http") and not url.startswith("https"):
            url = "http://{0}".format(url)
        parts = urlparse(url)
        _query = frozenset(parse_qsl(parts.query))
        _path = unquote_plus(parts.path)
        parts = parts._replace(query=_query, path=_path)
        domain = parts.netloc.strip("www.")
        return domain

    def get_source(self, url):

        if not url or not len(url):
            return None

        norm_url = self.normalize_url(url)
        if len(norm_url.split(".")) == 1:
            return url
        domain_name = ".".join(norm_url.split(".")[:-1])
        domain_name = domain_name.replace(".", "_")
        domain_name = domain_name.replace("/", "_")
        domain_name = domain_name.replace("\\", "_")
        return domain_name

    def clean_string(self, s):  
    
        # Remove all symbols and numbers from beginning and end of the string
        stripped_s = s.strip("@#$^&*-_+=[]{}|\\<>/\n")
        stripped_s = stripped_s.strip() # Stripping left-over whitespaces if any

        # Strip all types of bullet points
        pattern = r'^\s*(\•|\○|\*|\-|[0-9]+\.)\s*'
        stripped_s = re.sub(pattern, '', stripped_s)
        stripped_s = stripped_s.strip() # Stripping left-over whitespaces if any

        return stripped_s

    @staticmethod
    def remove_duplicate_string(strings_list):
        strings_df = pd.DataFrame({"text": strings_list})
        strings_df["hash_id"] = strings_df["text"].apply(lambda x: sha256(x.encode('utf-8')).hexdigest())
        deduped_strings_df = strings_df.drop_duplicates(subset=["hash_id"])
        return deduped_strings_df

    def split_into_chunks(self):
        chunks = [self.clean_string(chunk) for chunk in self.text.split("\n") if len(chunk)]
        return self.remove_duplicate_string(chunks)

    def replace_string(self, original, translated):
        self.translated_text = self.translated_text.replace(original, translated)

    @staticmethod
    def split_with_delimiter(
        text,
        # delimiter_pattern=r'[.?!।|॥؟۔](?:\n+)?'
        delimiter_pattern=r'(?<!\d)\.(?!\d)|(?<!\w)\.(?!\w)|[?!।|॥؟۔\n](?:\n+)?', 
    ):
        lines = re.split(f'({delimiter_pattern})', text)
        if len(lines) % 2 == 0:
            iter_range = range(0, len(lines), 2)
            out = [lines[i]+lines[i+1] for i in iter_range]
        else:
            iter_range = range(0, len(lines) - 1, 2)
            out = [lines[i]+lines[i+1] for i in iter_range] + [lines[-1]]
        return out

    def split_into_sentences(self, method="regex"):
        split_methods = {
            "regex": self.split_with_delimiter,
        }
        sents = [self.clean_string( sent.text if not isinstance(sent, str) else sent ) for sent in split_methods[method](self.text) if len(sent)]
        sents = [sent for sent in sents if len(sent)]
        return self.remove_duplicate_string(sents)

    @classmethod
    @staticmethod
    def get_template_doc_schema():
        return {
            "doc_id": [str(None)], 
            "source": [str(None)], 
            "url": [str(None)], 
            "timestamp": [str(None)], 
            "text": [str(None)], 
            "sub_strs": [str(None)], 
            "sids": [str(None)]
        }

    def get_templated_document_attrs(self):
        if self.translation_type == "chunk":
            df = self.chunks
        elif self.translation_type == "sentence":
            df = self.sentences

        if not self.timestamp:
            timestamp = str(None)        
        elif isinstance(self.timestamp, str):
            timestamp = self.timestamp  
        else:
            timestamp = self.timestamp.strftime("%m/%d/%Y, %H:%M:%S")

        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "url": self.url,
            "timestamp": timestamp,
            "text": self.text,
            "sub_strs": json.dumps(df["text"].tolist()),
            "sids": json.dumps(df["hash_id"].tolist()),
        }
    