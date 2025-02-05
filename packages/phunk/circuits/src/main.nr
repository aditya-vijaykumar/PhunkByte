use std::hash::poseidon2;

/// The main circuit function computes a primary hash from a 4-element input array
/// and derives a nullifier from the primary hash and a nonce.
/// It asserts that both the computed hash and nullifier match their expected public values.
fn main(
    inputs: [Field; 4],
    nonce: Field,
    expected_hash: pub Field,
    expected_nullifier: pub Field
) {
    // Validate that the input array is not entirely zero.
    let mut non_zero_found = false;
    for i in 0..inputs.len() {
        if inputs[i] != 0 {
            non_zero_found = true;
        }
    }
    assert(non_zero_found, "Input array is null (all zeros)");

    // Compute the primary hash using Poseidon2 over all 4 inputs.
    let computed_hash = poseidon2::Poseidon2::hash(inputs, inputs.len());
    assert(computed_hash != 0, "Computed hash is zero (nullified)");

    // Derive a nullifier from the computed hash and nonce.
    // We pack the computed hash and nonce into a 4-element array (padding with zeros)
    // and hash only the first two elements.
    let computed_nullifier = poseidon2::Poseidon2::hash([computed_hash, nonce, 0, 0], 2);

    // Verify that the computed hash and nullifier match the expected public values.
    assert(computed_hash == expected_hash, "Computed hash does not match expected value");
    assert(computed_nullifier == expected_nullifier, "Computed nullifier does not match expected value");
}

#[test]
fn test_valid_inputs() {
    // Define valid nonzero inputs and a nonce.
    let inputs: [Field; 4] = [1, 2, 3, 4];
    let nonce: Field = 5;
    // Compute the expected outputs using the same hash functions.
    let expected_hash = poseidon2::Poseidon2::hash(inputs, inputs.len());
    let expected_nullifier = poseidon2::Poseidon2::hash([expected_hash, nonce, 0, 0], 2);
    // Execute the circuit with the valid witness values.
    main(inputs, nonce, expected_hash, expected_nullifier);
}

#[test(should_fail_with = "Input array is null")]
fn test_all_zero_inputs_should_fail() {
    // An input array of all zeros should trigger input validation.
    let inputs: [Field; 4] = [0, 0, 0, 0];
    let nonce: Field = 5;
    // The expected outputs here are arbitrary since the circuit should fail before using them.
    main(inputs, nonce, 0, 0);
}

#[test(should_fail_with = "Computed hash does not match expected value")]
fn test_incorrect_expected_hash_should_fail() {
    // Use valid inputs but provide an incorrect expected hash.
    let inputs: [Field; 4] = [1, 2, 3, 4];
    let nonce: Field = 5;
    let computed_hash = poseidon2::Poseidon2::hash(inputs, inputs.len());
    let computed_nullifier = poseidon2::Poseidon2::hash([computed_hash, nonce, 0, 0], 2);
    // Deliberately alter the expected hash.
    let wrong_hash = computed_hash + 1;
    main(inputs, nonce, wrong_hash, computed_nullifier);
}

#[test(should_fail_with = "Computed nullifier does not match expected value")]
fn test_incorrect_expected_nullifier_should_fail() {
    // Use valid inputs but provide an incorrect expected nullifier.
    let inputs: [Field; 4] = [1, 2, 3, 4];
    let nonce: Field = 5;
    let computed_hash = poseidon2::Poseidon2::hash(inputs, inputs.len());
    let computed_nullifier = poseidon2::Poseidon2::hash([computed_hash, nonce, 0, 0], 2);
    // Deliberately alter the expected nullifier.
    let wrong_nullifier = computed_nullifier + 1;
    main(inputs, nonce, computed_hash, wrong_nullifier);
}

#[test]
fn test_hash_consistency() {
    // Verify that hashing the same inputs twice yields the same result.
    let inputs: [Field; 4] = [7, 8, 9, 10];
    let hash1 = poseidon2::Poseidon2::hash(inputs, inputs.len());
    let hash2 = poseidon2::Poseidon2::hash(inputs, inputs.len());
    assert(hash1 == hash2, "Hash function is not consistent for the same inputs");
}

#[test]
fn test_nullifier_uniqueness() {
    // Verify that different nonces yield different nullifiers for the same input.
    let inputs: [Field; 4] = [1, 3, 5, 7];
    let nonce1: Field = 9;
    let nonce2: Field = 10;
    let computed_hash = poseidon2::Poseidon2::hash(inputs, inputs.len());
    let nullifier1 = poseidon2::Poseidon2::hash([computed_hash, nonce1, 0, 0], 2);
    let nullifier2 = poseidon2::Poseidon2::hash([computed_hash, nonce2, 0, 0], 2);
    assert(nullifier1 != nullifier2, "Nullifiers must differ for different nonces");
}
