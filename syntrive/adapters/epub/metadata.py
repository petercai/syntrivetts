import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EpubMetadataProcessor:
    def __init__(self):
        pass
    
    def get_ebook_title(self, epubBook, first_doc):
        meta_title = epubBook.get_metadata("DC", "title")
        if meta_title and meta_title[0][0].strip():
            return meta_title[0][0].strip()
        if first_doc:
            from bs4 import BeautifulSoup
            html = first_doc.get_content().decode("utf-8")
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.select_one("head > title")
            if title_tag and title_tag.text.strip():
                return title_tag.text.strip()
            img = soup.find("img", alt=True)
            if img:
                alt = img['alt'].strip()
                if alt and "cover" not in alt.lower():
                    return alt
        return None
    
    def extract_book_images(self, epub_book, output_dir):
        import ebooklib
        from ebooklib import epub
        
        if isinstance(epub_book, str):
            if not os.path.exists(epub_book):
                raise FileNotFoundError(f"EPUB file not found: {epub_book}")
            
            try:
                epub_book = epub.read_epub(epub_book)
            except Exception as e:
                raise Exception(f"Failed to read EPUB file: {e}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_images = []
        
        for item in epub_book.get_items_of_type(ebooklib.ITEM_IMAGE):
            file_name = os.path.basename(item.file_name or "")
            
            if not file_name:
                file_name = f"image_{len(extracted_images)}.jpg"
            
            output_path = os.path.join(output_dir, file_name)
            
            with open(output_path, "wb") as out_file:
                out_file.write(item.get_content())
            
            thumbnail_path = None
                
            extracted_images.append((file_name, output_path, thumbnail_path))
        
        return extracted_images
    
    def extract_book_cover(self, epubBook, process_path, cover_name):        
        import ebooklib
        import io
        from PIL import Image

        try:
            cover_image = None
            cover_path = os.path.join(process_path, cover_name + '.jpg')
            cover_items = epubBook.get_items_of_type(ebooklib.ITEM_COVER)
            for item in cover_items:
                cover_image = item.get_content()
                break
            if not cover_image:
                for item in epubBook.get_items_of_type(ebooklib.ITEM_IMAGE):
                    if 'cover' in item.file_name.lower() or 'cover' in item.get_id().lower():
                        cover_image = item.get_content()
                        break
            if cover_image:
                image = Image.open(io.BytesIO(cover_image))
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                image.save(cover_path, format='JPEG')
                return cover_path
            return None
        except Exception:
            logger.exception("extract_book_cover() failed for cover_name=%r", cover_name)
            return None

    def extract_metadata(self, file_path: str) -> Dict[str, Optional[str]]:
        metadata = {
            'title': 'Unknown',
            'author': 'Unknown',
            'language': 'Unknown',
            'publisher': 'Unknown',
            'date': 'Unknown',
            'cover': None
        }
        
        if not file_path or not os.path.exists(file_path):
            return metadata
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.epub':
                metadata = self._extract_epub_metadata(file_path, metadata)
            else:
                metadata['title'] = Path(file_path).stem
        
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {e}")
            metadata['title'] = Path(file_path).stem if file_path else 'Unknown'
        
        return metadata
    
    def _extract_epub_metadata(
        self, 
        file_path: str, 
        metadata: Dict[str, Optional[str]]
    ) -> Dict[str, Optional[str]]:
        try:
            from ebooklib import epub
            
            book = epub.read_epub(file_path)
            
            title = book.get_metadata('DC', 'title')
            if title:
                metadata['title'] = title[0][0] if isinstance(title[0], tuple) else str(title[0])
            else:
                metadata['title'] = Path(file_path).stem
            
            creators = book.get_metadata('DC', 'creator')
            if creators:
                authors = [c[0] if isinstance(c, tuple) else str(c) for c in creators]
                metadata['author'] = ', '.join(authors)
            
            language = book.get_metadata('DC', 'language')
            if language:
                metadata['language'] = language[0][0] if isinstance(language[0], tuple) else str(language[0])
            
            publisher = book.get_metadata('DC', 'publisher')
            if publisher:
                metadata['publisher'] = publisher[0][0] if isinstance(publisher[0], tuple) else str(publisher[0])
            
            date = book.get_metadata('DC', 'date')
            if date:
                metadata['date'] = date[0][0] if isinstance(date[0], tuple) else str(date[0])
            
            cover_processor = CoverProcessor()
            metadata['cover'] = cover_processor.extract_cover(book, file_path)
        
        except ImportError:
            metadata['title'] = Path(file_path).stem
        
        except Exception as e:
            print(f"Error extracting EPUB metadata: {e}")
            metadata['title'] = Path(file_path).stem
        
        return metadata


class CoverProcessor:
    def __init__(self):
        pass
    
    def extract_cover(self, book, file_path: str) -> Optional[str]:
        try:
            import ebooklib
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_COVER:
                    return self._save_cover_image(item, file_path)
            
            for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
                if 'cover' in item.get_name().lower():
                    return self._save_cover_image(item, file_path)
        
        except Exception as e:
            print(f"Error extracting cover: {e}")
        
        return None
    
    def _save_cover_image(self, item, file_path: str) -> Optional[str]:
        try:
            cover_dir = tempfile.gettempdir()
            
            cover_filename = f"cover_{Path(file_path).stem}.jpg"
            cover_path = os.path.join(cover_dir, cover_filename)
            
            with open(cover_path, 'wb') as f:
                f.write(item.get_content())
            
            return cover_path
        
        except Exception as e:
            print(f"Error saving cover image: {e}")
            return None
