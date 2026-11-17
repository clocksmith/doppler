"""Bounded public matrix publication only, no model or cryptographic claims."""
from pathlib import Path
import tempfile
import threading
import unittest

class PublicColumnTests(unittest.TestCase):
    def test_same_digest_as_whole_matrix_and_create_only_publication(self):
        import numpy as np
        from public_weights import store_public_matrix,store_public_matrix_columns
        for bits,dtype in ((8,'int8'),(12,'<i2')):
            with self.subTest(bits=bits),tempfile.TemporaryDirectory() as directory:
                matrix=np.arange(35,dtype=dtype).reshape(5,7)-15
                parameters=dict(quantizationBits=bits,weightEncoding='signed-int8' if bits==8 else 'signed-int16-le')
                # Use the installed encoding identifier rather than inventing one.
                from weight_encoding import encoding_for_bits
                parameters['weightEncoding']=encoding_for_bits(bits)['id']
                whole,row=store_public_matrix(directory,'a',5,7,parameters,matrix.tobytes())
                inode=whole.stat().st_ino
                path,actual=store_public_matrix_columns(directory,'a',5,7,parameters,[matrix[:,:3],matrix[:,3:]])
                self.assertEqual(actual,row);self.assertEqual(path.stat().st_ino,inode)
                raw=whole.read_bytes();whole.write_bytes(bytes([raw[0]^1])+raw[1:])
                with self.assertRaisesRegex(ValueError,'corruption'):
                    store_public_matrix_columns(directory,'a',5,7,parameters,[matrix])
                self.assertEqual(whole.read_bytes(),bytes([raw[0]^1])+raw[1:])

    def test_failure_cleanup_preserves_unrelated_files(self):
        import numpy as np
        from public_weights import store_public_matrix_columns
        from weight_encoding import encoding_for_bits
        parameters=dict(quantizationBits=12,weightEncoding=encoding_for_bits(12)['id'])
        for failure in ('short','long','dtype','cancel'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                sibling=Path(directory)/'keep';sibling.write_text('keep');event=threading.Event()
                def columns():
                    if failure=='cancel':event.set()
                    yield np.zeros((5,8 if failure=='long' else 3),dtype='int8' if failure=='dtype' else '<i2')
                with self.assertRaises((ValueError,InterruptedError)):
                    store_public_matrix_columns(directory,'a',5,7,parameters,columns(),cancelled=event)
                self.assertEqual(list(Path(directory).iterdir()),[sibling])

if __name__=='__main__':unittest.main()
