#include <algorithm>
#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <unistr.h>
#include <unicode/normalizer2.h>
//#include <unicode/schriter.h>
#include <unicode/locid.h>
#include <unicode/brkiter.h>
#include <math.h>

using namespace tensorflow;


REGISTER_OP("SplitTokens")
  .Input("source: string")
  .Output("indices: int64")
  .Output("values: string")
  .Output("shape: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused)); // source

    c->set_output(0, c->Matrix(shape_inference::InferenceContext::kUnknownDim, 2)); // indices
    c->set_output(1, c->Vector(shape_inference::InferenceContext::kUnknownDim)); // values
    c->set_output(2, c->Vector(2)); // shape

    return Status::OK();
  })
  .SetIsStateful();


class SplitTokensOp : public OpKernel {
 public:
  explicit SplitTokensOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Create Normalizer2 instance
    UErrorCode nfcError = U_ZERO_ERROR;
    nfcNormalizer = Normalizer2::getNFCInstance(nfcError);
    OP_REQUIRES(ctx, U_SUCCESS(nfcError), errors::InvalidArgument("Normalizer2 instantiation failed"));

    // Create word-level BreakIterator instance
    UErrorCode wordError = U_ZERO_ERROR;
    wordIterator = BreakIterator::createWordInstance(Locale::getRoot(), wordError);
    OP_REQUIRES(ctx, U_SUCCESS(wordError), errors::InvalidArgument("BreakIterator instantiation failed"));


//    char buffer[500];
//
//    for (UChar32 i = 0; i < 0x100; i++) {
////    for (int64 i = 0; i < 0x10ffff; i++) {
//      // U_UNICODE_CHAR_NAME - U_EXTENDED_CHAR_NAME - U_CHAR_NAME_ALIAS
//      UErrorCode charError = U_ZERO_ERROR;
//      int length = u_charName(i, U_UNICODE_10_CHAR_NAME, buffer, 500, &charError);
//      std::cout << buffer << std::endl;
//
//      OP_REQUIRES(ctx, U_SUCCESS(charError), errors::InvalidArgument("Error retrieving character name"));
//    }
  }


  void Compute(OpKernelContext* ctx) override {
    // Prepare input
    const Tensor* source_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source", &source_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(source_tensor->shape()),
      errors::InvalidArgument("source must be a vector, got shape: ", source_tensor->shape().DebugString())
    );

    const auto source_vec = source_tensor->vec<string>();
    const int64 batch_size = source_vec.dimension(0);


    // Estimate approximate result size
    int64 space_count = 0;
    for (int64 i = 0; i < batch_size; i++) {
      space_count += std::count(source_vec(i).begin(), source_vec(i).end(), ' ');
    }


    // Result tokens, their rows & sizes
    std::vector<UnicodeString> result_tokens;
    result_tokens.reserve((space_count + batch_size) * 4);

    std::vector<int64> result_rows;
    result_rows.reserve((space_count + batch_size) * 4);

    std::vector<int64> result_sizes;
    result_sizes.reserve(batch_size);


    // Split source strings
    for (int64 i = 0; i < batch_size; i++) {
      string source_string = source_vec(i);

      // Decode from UTF-8
      UnicodeString unicode_string;
      UErrorCode decodeError = U_ZERO_ERROR;
      decode(source_string, unicode_string, decodeError);
      OP_REQUIRES(ctx, U_SUCCESS(decodeError), errors::InvalidArgument("unicode decoding failed"));

      // Split to words
      int64 tokens_count = result_tokens.size(); // Remember old size
      words(unicode_string, result_tokens);
      tokens_count = result_tokens.size() - tokens_count; // Estimate size change
      
      // Remember row binding and row size
      result_rows.insert(result_rows.end(), tokens_count, i);
      result_sizes.push_back(tokens_count);
    }


    // Result metrics
    int64 output_size = result_tokens.size();
    int64 max_values = (result_sizes.size() > 0) ? *std::max_element(result_sizes.begin(), result_sizes.end()) : 0;


    // Allocate output
    Tensor* indices_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}), &indices_tensor));
    Tensor* values_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({output_size}), &values_tensor));
    Tensor* shape_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &shape_tensor));

    auto indices_matrix = indices_tensor->matrix<int64>();
    auto values_vec = values_tensor->vec<string>();
    auto shape_vec = shape_tensor->vec<int64>();


    // Store shape
    shape_vec(0) = batch_size;
    shape_vec(1) = max_values;

    // Store values & indices
    int64 prev_row = 0;
    int64 curr_col = 0;
    for (int64 i = 0; i < output_size; i++) {
      // Encode to UTF-8
      string result_value;
      encode(result_tokens[i], result_value);
      values_vec(i) = result_value;

      // Write indices
      if (result_rows[i] > prev_row) {
        curr_col = 0;
      }

      indices_matrix(i, 0) = result_rows[i];
      indices_matrix(i, 1) = curr_col;

      prev_row = result_rows[i];
      curr_col++;
    }
  }
 private:
  mutex nfcMutex;
  const Normalizer2 *nfcNormalizer GUARDED_BY(nfcMutex);

  mutex wordMutex;
  BreakIterator *wordIterator GUARDED_BY(wordMutex);

  
  // Decode from Bytes to Unicode
  void decode(const string &source, UnicodeString &target, UErrorCode &error) {
    // TODO: check for correct UTF-8?
    // http://bjoern.hoehrmann.de/utf-8/decoder/dfa/

    // Decode from UTF-8
    target = UnicodeString::fromUTF8(source);

    // Convert to NFC-form
    target = nfcNormalizer->normalize(target, error);
  }

  // Encode from Unicode to Bytes
  void encode(const UnicodeString &source, string &target) {
    target.clear();
    source.toUTF8String(target);
  }

  // Split string to tokens
  void words(const UnicodeString &source, std::vector<UnicodeString> &target) {
    // Preserve empty token and exit fast
    if (0 == source.length()) {
      target.push_back("");

      return;
    }

    std::vector<int32_t> positions;

    // Split words by Unicode rules
    wordIterator->setText(source);
    for (int32_t pos = wordIterator->first(); pos != BreakIterator::DONE; pos = wordIterator->next()) {
      positions.push_back(pos);
    }

    // Split words by punctuation code points
    for (int32_t pos = 0; pos < source.length() - 1; pos++) {
      UChar unit = source.charAt(pos);

      if (u_ispunct(unit)) {
        positions.push_back(pos);
        positions.push_back(pos + 1);
      }
    }

    // TODO: check if joining '.', '.', '.' to '...' makes predictions better

    // Remove duplicate positions
    std::sort(positions.begin(), positions.end());
    auto last_pos = std::unique(positions.begin(), positions.end());
    positions.erase(last_pos, positions.end());

    for (int64 i = 0; i < positions.size() - 1; i++) {
      UnicodeString word = UnicodeString(source, positions[i], positions[i+1] - positions[i]);
      target.push_back(word);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SplitTokens").Device(DEVICE_CPU), SplitTokensOp);


REGISTER_OP("ExtractFeatures")
  .Input("in_indices: int64")
  .Input("in_values: string")
  .Input("in_shape: int64")
  .Output("out_indices: int64")
  .Output("out_values: float")
  .Output("out_shape: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused)); // in_indices
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused)); // in_values
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused)); // in_shape

    c->set_output(0, c->Matrix(shape_inference::InferenceContext::kUnknownDim, 3)); // out_indices
    c->set_output(1, c->Vector(shape_inference::InferenceContext::kUnknownDim)); // out_values
    c->set_output(2, c->Vector(3)); // out_shape

    return Status::OK();
  })
  .SetIsStateful();


class ExtractFeaturesOp : public OpKernel {
 public:
  explicit ExtractFeaturesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }


  void Compute(OpKernelContext* ctx) override {
    // Prepare input
    const Tensor* in_indices_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("in_indices", &in_indices_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(in_indices_tensor->shape()),
      errors::InvalidArgument("in_indices must be a matrix, got shape: ", in_indices_tensor->shape().DebugString())
    );

    const Tensor* in_values_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("in_values", &in_values_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(in_values_tensor->shape()),
      errors::InvalidArgument("in_values must be a vector, got shape: ", in_values_tensor->shape().DebugString())
    );

    const Tensor* in_shape_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("in_shape", &in_shape_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(in_shape_tensor->shape()),
      errors::InvalidArgument("in_shape must be a vector, got shape: ", in_shape_tensor->shape().DebugString()));
    OP_REQUIRES(ctx, in_shape_tensor->shape().dim_size(0) == 2,
      errors::InvalidArgument("in_shape must be a vector with 2 values (batch size, max values), got size: ", in_shape_tensor->shape().DebugString()));

    const auto in_indices_matrix = in_indices_tensor->matrix<int64>();
    const auto in_values_vector = in_values_tensor->vec<string>();
    const auto in_shape_vector = in_shape_tensor->vec<int64>();

    const int64 total_tokens_count = in_values_vector.dimension(0);


    // Extract features
    std::vector<float> all_features;
    all_features.reserve(total_tokens_count * features_per_token);

    std::vector<float> sample_features;
    sample_features.reserve(features_per_token);


    for (int64 i = 0; i < total_tokens_count; i++) {
      UnicodeString sample_token = UnicodeString::fromUTF8(in_values_vector(i));

      sample_features.clear();
      featureCase(sample_token, sample_features);
      featureLength(sample_token, sample_features);
      featureCharTypes(sample_token, sample_features);
      featureBeginEnd(sample_token, sample_features);
      featureCharSound(sample_token, sample_features);

      all_features.insert(all_features.end(), sample_features.begin(), sample_features.end());
    }
    OP_REQUIRES(ctx, all_features.size() == total_tokens_count * features_per_token,
      errors::InvalidArgument("total features count does not match expected one"));


    // Evaluate non-zero features count
    int64 meaning_features_size = 0;
    for (float value : all_features )
    {
      if (0 == value) {
        continue;
      }
      meaning_features_size++;
    }


    // Allocate output
    Tensor* out_indices_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({meaning_features_size, 3}), &out_indices_tensor));
    auto out_indices_matrix = out_indices_tensor->matrix<int64>();

    Tensor* out_values_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({meaning_features_size}), &out_values_tensor));
    auto out_values_vector = out_values_tensor->vec<float>();

    Tensor* out_shape_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({3}), &out_shape_tensor));
    auto out_shape_vector = out_shape_tensor->vec<int64>();


    // Copy result to output
    int64 meaning_i = 0;
    for (int64 all_i = 0; all_i < all_features.size(); all_i++) {
      float feature_value = all_features[all_i];
      if (0 == feature_value) {
        continue;
      }

      int64 old_i = all_i / features_per_token;

      out_indices_matrix(meaning_i, 0) = in_indices_matrix(old_i, 0);
      out_indices_matrix(meaning_i, 1) = in_indices_matrix(old_i, 1);
      out_indices_matrix(meaning_i, 2) = all_i % features_per_token;

      out_values_vector(meaning_i) = feature_value;

      meaning_i++;
    }

    out_shape_vector(0) = in_shape_vector(0);
    out_shape_vector(1) = in_shape_vector(1);
    out_shape_vector(2) = features_per_token;
  }
 private:
  const int features_per_token = 36;
  const int max_token_length = 25;
  const std::set<UChar32> consonants_lower = {
    // Latin
    0x0062, 0x0063, 0x0064, 0x0066, 0x0067, 0x0068, 0x006A, 0x006B, 0x006C, 0x006D, 0x006E, 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0076, 0x0077, 0x0078, 0x007A,
    // Cyrillic
    0x0431, 0x0432, 0x0433, 0x0434, 0x0436, 0x0437, 0x043A, 0x043B, 0x043C, 0x043D, 0x043F, 0x0440, 0x0441, 0x0442, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449
  };
  const std::set<UChar32> vowels_lower = {
    // Latin
    0x0061, 0x0065, 0x0069, 0x006F, 0x0075,
    // Cyrillic
    0x0430, 0x0435, 0x0438, 0x043E, 0x0443, 0x044B, 0x044D, 0x044E, 0x044F, 0x0451
  };
  const std::set<UChar32> soundless_lower = {
    // Cyrillic
    0x044A,  // ъ
    0x044C // ь
  };
  const std::set<UChar32> exceptions_lower =  {
    // Latin
    0x0079, //  y
    // Cyrillic
    0x0439 // й
  };

  /**
    * Features taken from
    * 2005 - https://nlp.stanford.edu/courses/cs224n/2005/agarwal_herndon_shneider_final.pdf
    *   + is upper case, is title case
    *   + length
    *   - one-hot for common punctuation
    *   - one-hot for brackets and quotes
    *   - is open, is close flag for brackets and quotes
    *   + is number
    *   - is abbreviation
    *
    * 2012 - http://www.aclweb.org/anthology/C12-2096
    *   - probabilities of words being sentence-final or -initial
    *   + word length
    *   + word case
    *   - list of abbreviations
    *
    * 2013 - http://www.aclweb.org/anthology/D13-1146
    *   - POS
    *   + capitalization
    *   - is abbreviation
    *   - one-hot for each character
    *   - Unicode category
    *
    * 2014 - http://www.wellformedness.com/blog/simpler-sentence-boundary-detection/
    *   + is punctuation
    *   + contain a vowel
    *   - contain a period
    *   + length
    *   + case
    *
    * 2015 - http://amitavadas.com/Pub/SBD_ICON_2015.pdf
    *   - is emoticon
    *   - consecutive punctuation markers
    *   - sequences of multiple periods
    *   - punctuations in different combinations
    *   - POS probabilities
    *   - is abbreviation
    *   + length
    *   - contain internal period
    *   + case
    *   - status of $
    *   + is number
    *   - is email
    *   - is web address
    *   - unusual sentence end markers: |, ||, *
    *
    * 2018 - this model additional features
    *   + case: no, mixed
    *   + starts/ends with number/alpha
    *   + vowel/consonant/soundless/exceptional characters total/proportional count in word/ending
    */
  //ONE_HOT for 1-2 chars
  //PROBABILITY_START
  //PROBABILITY_END

  void featureCase(const UnicodeString &test, std::vector<float> &features) {
    UnicodeString upper(test);
    upper.toUpper();

    UnicodeString lower(test);
    lower.toLower();

    bool noCase = upper == lower;
    bool upperCase = false;
    bool lowerCase = false;
    bool titleCase = false;

    if (!noCase && test == upper) {
      upperCase = true;
    }
    if (!noCase && test == lower) {
      lowerCase = true;
    }
    if (!noCase && !test.compare(0, 1, upper, 0, 1) && !test.compare(1, test.length() - 1, lower, 1, lower.length() - 1)) {
      titleCase = true;
    }

    features.push_back(noCase);
    features.push_back(upperCase);
    features.push_back(lowerCase);
    features.push_back(titleCase);
    features.push_back(!noCase && !upperCase && !lowerCase && !titleCase); // Mixed case
  }

  void featureLength(const UnicodeString &test, std::vector<float> &features) {
    // Is empty
    features.push_back(0 == test.length());

    // Log10 for length
    features.push_back(0 == test.length() ? 0 : (float) test.length() / max_token_length);
  }

  void featureCharTypes(const UnicodeString &test, std::vector<float> &features) {
    int digit = 0;
    int alpha = 0;
    int punct = 0;
    int graph = 0;
    int blank = 0;
    int space = 0;
    int cntrl = 0;
    int print = 0;
    int base = 0;

    for (int32_t pos = 0; pos < test.length(); pos++) {
      UChar unit = test.charAt(pos);
      digit += u_isdigit(unit);
      alpha += u_isalpha(unit);
      punct += u_ispunct(unit);
      graph += u_isgraph(unit);
      blank += u_isblank(unit);
      space += u_isspace(unit);
      cntrl += u_iscntrl(unit);
      print += u_isprint(unit);
      base += u_isbase(unit);
    }

    int length = test.length() == 0 ? 1 : test.length();

    features.push_back((float)digit / length);
    features.push_back((float)alpha / length);
    features.push_back((float)punct / length);
    features.push_back((float)graph / length);
    features.push_back((float)blank / length);
    features.push_back((float)space / length);
    features.push_back((float)cntrl / length);
    features.push_back((float)print / length);
    features.push_back((float)base / length);
  }

  void featureBeginEnd(const UnicodeString &test, std::vector<float> &features) {
    int begin_alpha = false;
    int begin_number = false;
    int end_alpha = false;
    int end_number = false;

    if (test.length() > 0) {
      begin_alpha = u_isalpha(test.charAt(0));
      begin_number = u_isdigit(test.charAt(0));
      end_alpha = u_isalpha(test.charAt(test.length() - 1));
      end_number = u_isdigit(test.charAt(test.length() - 1));
    }

    features.push_back(begin_alpha);
    features.push_back(begin_number);
    features.push_back(end_alpha);
    features.push_back(end_number);
  }


  void featureCharSound(const UnicodeString &test, std::vector<float> &features) {
    int length = test.length() == 0 ? 1 : test.length();

    UnicodeString lower(test);
    lower.toLower();


    int consonants_total = 0;
    int vowels_total = 0;
    int soundless_total = 0;
    int exceptions_total = 0;

    for (int32_t pos = 0; pos < lower.length(); pos++) {
      UChar unit = lower.charAt(pos);

      consonants_total += consonants_lower.find(unit) != consonants_lower.end();
      vowels_total += vowels_lower.find(unit) != vowels_lower.end();
      soundless_total += soundless_lower.find(unit) != soundless_lower.end();
      exceptions_total += exceptions_lower.find(unit) != exceptions_lower.end();
    }

    // Proportional to absolute length
    features.push_back((float)consonants_total * 2 / max_token_length);
    features.push_back((float)vowels_total * 2 / max_token_length);
    features.push_back((float)soundless_total * 2 / max_token_length);
    features.push_back((float)exceptions_total * 2 / max_token_length);

    // Proportional to length
    features.push_back((float)consonants_total / length);
    features.push_back((float)vowels_total / length);
    features.push_back((float)soundless_total / length);
    features.push_back((float)exceptions_total / length);


    // At the end
    int consonants_end = 0;
    int vowels_end = 0;
    int soundless_end = 0;
    int exceptions_end = 0;

    bool consonants_break = false;
    bool vowels_break = false;
    bool soundless_break = false;
    bool exceptions_break = false;

    for (int32_t pos = lower.length() - 1; pos >= 0 ; pos--) {
      UChar unit = lower.charAt(pos);

      if (!consonants_break && consonants_lower.find(unit) != consonants_lower.end()) {
        consonants_end += 1;
      } else {
        consonants_break = true;
      }

      if (!vowels_break && vowels_lower.find(unit) != vowels_lower.end()) {
        vowels_end += 1;
      } else {
        vowels_break = true;
      }

      if (!soundless_break && soundless_lower.find(unit) != soundless_lower.end()) {
        soundless_end += 1;
      } else {
        soundless_break = true;
      }

      if (!exceptions_break && exceptions_lower.find(unit) != exceptions_lower.end()) {
        exceptions_end += 1;
      } else {
        exceptions_break = true;
      }
    }

    // Proportional to absolute length
    features.push_back((float)consonants_end * 5 / max_token_length);
    features.push_back((float)vowels_end * 5 / max_token_length);
    features.push_back((float)soundless_end * 5 / max_token_length);
    features.push_back((float)exceptions_end * 5 / max_token_length);

    // Proportional to length
    features.push_back((float)consonants_end / length);
    features.push_back((float)vowels_end / length);
    features.push_back((float)soundless_end / length);
    features.push_back((float)exceptions_end / length);
  }

//  void featureOneHot(const UnicodeString &test, std::vector<float> &features) {
//    // dot
//
//    // Is empty
//    features.push_back(0 == test.length());
//
//    // Log10 for length
//    features.push_back(0 == test.length() ? 0 : (float) test.length() / max_token_length);
//  }
//
//  void feature???(const UnicodeString &test, std::vector<float> &features) {
//    // ? UProperty
//    // + UCharCategory
//    // ? UCharDirection
//    // ? UBidiPairedBracketType
//    // ? UBlockCode
//    // USentenceBreak
//    // ULineBreak
//
//    // Is empty
//    features.push_back(0 == test.length());
//
//    // Log10 for length
//    features.push_back(0 == test.length() ? 0 : (float) test.length() / max_token_length);
//  }

//  u_charType -> http://icu-project.org/apiref/icu4c/uchar_8h.html#a6a2dbc531efce8d77fdb4c314e7fc25e
//UProperty ?
//UCharCategory
//UCharDirection ?
//UBidiPairedBracketType
//UBlockCode
//u_charName + UCharNameChoice -> http://icu-project.org/apiref/icu4c/uchar_8h.html#aa488f2a373998c7decb0ecd3e3552079
//u_getPropertyName()  + u_getPropertyValueName() + UPropertyNameChoice
};

REGISTER_KERNEL_BUILDER(Name("ExtractFeatures").Device(DEVICE_CPU), ExtractFeaturesOp);

