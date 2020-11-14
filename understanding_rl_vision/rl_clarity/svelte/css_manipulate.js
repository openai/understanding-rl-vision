export const css_manipulate = function(css_value, numeric_fn) {
  let last_digit_index = css_value.search(/\d(?!.*\d.*)/);
  if (last_digit_index === -1) {
    return css_value;
  }
  else {
    let numeric_value = parseFloat(css_value.substring(0, last_digit_index + 1));
    return numeric_fn(numeric_value).toString() + css_value.substring(last_digit_index + 1);
  }
};

export const css_multiply = function(css_value, multiplier) {
  return css_manipulate(css_value, function(numeric_value) {
    return numeric_value * multiplier;
  });
};
