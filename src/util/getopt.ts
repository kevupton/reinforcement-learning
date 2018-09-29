// syntactic sugar function for getting default parameter values
export function getopt (opt : any, field_name : string, default_value : any) {
  if(typeof opt === 'undefined') { return default_value; }
  return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
}
