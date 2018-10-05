// syntactic sugar function for getting default parameter values
import { AgentOptions } from '../lib/interfaces';

export function getopt<K extends keyof AgentOptions> (opt : AgentOptions, field_name : K, default_value : AgentOptions[K]) : AgentOptions[K] {
  if(typeof opt === 'undefined') { return default_value; }
  return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
}
