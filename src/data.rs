use std::collections::{BTreeSet, HashMap};

pub struct Vocab {
    pub tokens: Vec<String>,
    pub stoi: HashMap<String, usize>,
}

impl Vocab {
    pub fn bos(&self) -> usize {
        self.stoi["<BOS>"]
    }
    pub fn eos(&self) -> usize {
        self.stoi["<EOS>"]
    }
    pub fn size(&self) -> usize {
        self.tokens.len()
    }
}

pub fn build_vocab(docs: &[&str]) -> Vocab {
    let mut chars = BTreeSet::new();
    for d in docs {
        for c in d.chars() {
            chars.insert(c);
        }
    }
    let mut tokens = vec!["<BOS>".to_string(), "<EOS>".to_string()];
    tokens.extend(chars.iter().map(|c| c.to_string()));
    let stoi: HashMap<String, usize> = tokens
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    Vocab { tokens, stoi }
}

pub fn tokenize(doc: &str, vocab: &Vocab, block_size: usize) -> Vec<usize> {
    std::iter::once(vocab.bos())
        .chain(doc.chars().map(|c| vocab.stoi[&c.to_string()]))
        .chain(std::iter::once(vocab.eos()))
        .take(block_size)
        .collect()
}
