use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::inference::generate;
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::train::train_step;

fn full_dataset() -> Vec<&'static str> {
    vec![
        "emma",
        "olivia",
        "ava",
        "isabella",
        "sophia",
        "mia",
        "charlotte",
        "amelia",
        "harper",
        "evelyn",
        "liam",
        "noah",
        "oliver",
        "elijah",
        "william",
        "james",
        "benjamin",
        "lucas",
        "mason",
        "ethan",
        "aiden",
        "logan",
        "jackson",
        "sebastian",
        "mateo",
        "jack",
        "owen",
        "theodore",
        "samuel",
        "henry",
        "leo",
        "luke",
        "jayden",
        "gabriel",
        "landon",
        "anthony",
        "dylan",
        "carter",
        "julian",
        "layla",
        "zoe",
        "penelope",
        "lily",
        "eleanor",
        "nora",
        "luna",
        "hazel",
        "aurora",
        "chloe",
        "aria",
        "grace",
        "zoey",
        "riley",
        "violet",
        "nova",
        "camille",
        "claire",
        "isla",
        "sofia",
        "scarlett",
        "elena",
        "alice",
        "savannah",
        "daisy",
        "audrey",
        "ruby",
        "stella",
        "naomi",
        "adeline",
        "ryan",
        "caleb",
        "eli",
        "christian",
        "josiah",
        "nathan",
        "wyatt",
        "andrew",
        "joshua",
        "christopher",
        "lincoln",
        "thomas",
        "ezra",
        "hudson",
        "daniel",
        "nicholas",
        "peter",
        "john",
        "levi",
        "ian",
        "axel",
        "cole",
        "beau",
        "felix",
        "maya",
        "nadia",
        "iris",
        "june",
        "vera",
    ]
}

#[test]
fn param_count_matches_original() {
    let mc = ModelConfig::default();
    let mut rng = Rng::new(42);
    let docs = full_dataset();
    let vocab = build_vocab(&docs);
    assert_eq!(vocab.size(), 27);
    let model = Model::new(vocab.size(), &mut rng, mc);
    assert_eq!(model.param_count(), 3632);
}

#[test]
fn training_reduces_loss() {
    let mc = ModelConfig::default();
    let tc = TrainConfig {
        n_steps: 200,
        ..TrainConfig::default()
    };
    let mut rng = Rng::new(42);
    let docs = vec!["emma", "olivia", "liam", "noah"];
    let vocab = build_vocab(&docs);
    let mut model = Model::new(vocab.size(), &mut rng, mc);

    let tokens = tokenize("emma", &vocab, mc.block_size);
    let loss_first = train_step(&mut model, &tokens, 0, &tc);

    for step in 1..200 {
        let doc = docs[step % docs.len()];
        let toks = tokenize(doc, &vocab, mc.block_size);
        train_step(&mut model, &toks, step, &tc);
    }

    let loss_last = train_step(&mut model, &tokens, 200, &tc);
    assert!(
        loss_last < loss_first,
        "Loss should decrease: {loss_first} -> {loss_last}"
    );
}

#[test]
fn generate_produces_output() {
    let mc = ModelConfig::default();
    let mut rng = Rng::new(42);
    let docs = full_dataset();
    let vocab = build_vocab(&docs);
    let model = Model::new(vocab.size(), &mut rng, mc);
    let samples = generate(&model.w, &vocab, &mut rng, &mc, 3);
    assert_eq!(samples.len(), 3);
    for s in &samples {
        assert!(!s.is_empty(), "Generated name should not be empty");
    }
}
