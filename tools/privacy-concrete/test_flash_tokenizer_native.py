"""Actual pinned Flash vocabulary on public text, not a model execution test."""
import os
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_TOKENIZER_DIR'),'opt-in exact public Flash tokenizer files')
class ActualFlashTokenizerTests(unittest.TestCase):
    def test_actual_public_completions_roundtrip_with_the_fixed_fast_implementation(self):
        from flash_tokenizer import encode,decode
        directory=os.environ['DOPPLER_FLASH_TOKENIZER_DIR']
        for prompt in ['The capital of France is','Hello, world!']:
            tokens=encode(directory,prompt,8)
            self.assertGreater(len(tokens),0);self.assertLessEqual(len(tokens),8)
            self.assertEqual(decode(directory,tokens,8),prompt)
        with self.assertRaises(ValueError):encode(directory,'The capital of France is',1)


if __name__=='__main__':unittest.main()
