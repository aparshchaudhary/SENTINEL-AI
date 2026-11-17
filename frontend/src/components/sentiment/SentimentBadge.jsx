import React from 'react';
import { Smile, Frown, MinusCircle } from 'lucide-react';
import Badge from '../common/Badge';

export const SentimentBadge = ({ sentiment, size = 'md', showIcon = true, className = '' }) => {
  const norm = (sentiment || '').toLowerCase();

  let icon = null;
  let variant = 'slate';
  let label = 'Unknown';

  if (norm === 'positive') {
    variant = 'positive';
    label = 'Positive';
    icon = <Smile className="w-3.5 h-3.5 mr-1 shrink-0" />;
  } else if (norm === 'negative') {
    variant = 'negative';
    label = 'Negative';
    icon = <Frown className="w-3.5 h-3.5 mr-1 shrink-0" />;
  } else if (norm === 'neutral') {
    variant = 'neutral';
    label = 'Neutral';
    icon = <MinusCircle className="w-3.5 h-3.5 mr-1 shrink-0" />;
  }

  return (
    <Badge variant={variant} size={size} dot={!showIcon} className={className}>
      {showIcon && icon}
      {label}
    </Badge>
  );
};

export default SentimentBadge;
