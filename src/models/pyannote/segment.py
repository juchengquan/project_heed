#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2021 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from typing import Optional, Iterator, Literal

Alignment = Literal['center', 'loose', 'strict']

import numpy as np
from dataclasses import dataclass

# setting 'frozen' to True makes it hashable and immutable
@dataclass(frozen=True, order=True)
class Segment:
    """
    Time interval

    Parameters
    ----------
    start : float
        interval start time, in seconds.
    end : float
        interval end time, in seconds.


    Segments can be compared and sorted using the standard operators:

    >>> Segment(0, 1) == Segment(0, 1.)
    True
    >>> Segment(0, 1) != Segment(3, 4)
    True
    >>> Segment(0, 1) < Segment(2, 3)
    True
    >>> Segment(0, 1) < Segment(0, 2)
    True
    >>> Segment(1, 2) < Segment(0, 3)
    False

    Note
    ----
    A segment is smaller than another segment if one of these two conditions is verified:

      - `segment.start < other_segment.start`
      - `segment.start == other_segment.start` and `segment.end < other_segment.end`

    """
    start: float = 0.0
    end: float = 0.0

    @staticmethod
    def set_precision(ndigits: Optional[int] = None):
        """Automatically round start and end timestamps to `ndigits` precision after the decimal point

        To ensure consistency between `Segment` instances, it is recommended to call this method only 
        once, right after importing `pyannote.core.Segment`.

        Usage
        -----
        >>> from pyannote.core import Segment
        >>> Segment.set_precision(2)
        >>> Segment(1/3, 2/3)
        <Segment(0.33, 0.67)>
        """
        global AUTO_ROUND_TIME
        global SEGMENT_PRECISION

        if ndigits is None:
            # backward compatibility
            AUTO_ROUND_TIME = False
            # 1 μs (one microsecond)
            SEGMENT_PRECISION = 1e-6
        else:
            AUTO_ROUND_TIME = True
            SEGMENT_PRECISION = 10 ** (-ndigits)

    def __bool__(self):
        """Emptiness

        >>> if segment:
        ...    # segment is not empty.
        ... else:
        ...    # segment is empty.

        Note
        ----
        A segment is considered empty if its end time is smaller than its
        start time, or its duration is smaller than 1μs.
        """
        return bool((self.end - self.start) > SEGMENT_PRECISION)

    def __post_init__(self):
        """Round start and end up to SEGMENT_PRECISION precision (when required)"""
        if AUTO_ROUND_TIME:
            object.__setattr__(self, 'start', int(self.start / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION)
            object.__setattr__(self, 'end', int(self.end / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION)

    @property
    def duration(self) -> float:
        """Segment duration (read-only)"""
        return self.end - self.start if self else 0.

    @property
    def middle(self) -> float:
        """Segment mid-time (read-only)"""
        return .5 * (self.start + self.end)

    def __iter__(self) -> Iterator[float]:
        """Unpack segment boundaries
        >>> segment = Segment(start, end)
        >>> start, end = segment
        """
        yield self.start
        yield self.end

    def copy(self) -> 'Segment':
        """Get a copy of the segment

        Returns
        -------
        copy : Segment
            Copy of the segment.
        """
        return Segment(start=self.start, end=self.end)

    # ------------------------------------------------------- #
    # Inclusion (in), intersection (&), union (|) and gap (^) #
    # ------------------------------------------------------- #

    def __contains__(self, other: 'Segment'):
        """Inclusion

        >>> segment = Segment(start=0, end=10)
        >>> Segment(start=3, end=10) in segment:
        True
        >>> Segment(start=5, end=15) in segment:
        False
        """
        return (self.start <= other.start) and (self.end >= other.end)

    def __and__(self, other):
        """Intersection

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment & other_segment
        <Segment(5, 10)>

        Note
        ----
        When the intersection is empty, an empty segment is returned:

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> intersection = segment & other_segment
        >>> if not intersection:
        ...    # intersection is empty.
        """
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Segment(start=start, end=end)

    def intersects(self, other: 'Segment') -> bool:
        """Check whether two segments intersect each other

        Parameters
        ----------
        other : Segment
            Other segment

        Returns
        -------
        intersect : bool
            True if segments intersect, False otherwise
        """

        return (self.start < other.start and
                other.start < self.end - SEGMENT_PRECISION) or \
               (self.start > other.start and
                self.start < other.end - SEGMENT_PRECISION) or \
               (self.start == other.start)

    def overlaps(self, t: float) -> bool:
        """Check if segment overlaps a given time

        Parameters
        ----------
        t : float
            Time, in seconds.

        Returns
        -------
        overlap: bool
            True if segment overlaps time t, False otherwise.
        """
        return self.start <= t and self.end >= t

    def __or__(self, other: 'Segment') -> 'Segment':
        """Union

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment | other_segment
        <Segment(0, 15)>

        Note
        ----
        When a gap exists between the segment, their union covers the gap as well:

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment | other_segment
        <Segment(0, 20)
        """

        # if segment is empty, union is the other one
        if not self:
            return other
        # if other one is empty, union is self
        if not other:
            return self

        # otherwise, do what's meant to be...
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        return Segment(start=start, end=end)

    def __xor__(self, other: 'Segment') -> 'Segment':
        """Gap

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment ^ other_segment
        <Segment(10, 15)

        Note
        ----
        The gap between a segment and an empty segment is not defined.

        >>> segment = Segment(0, 10)
        >>> empty_segment = Segment(11, 11)
        >>> segment ^ empty_segment
        ValueError: The gap between a segment and an empty segment is not defined.
        """

        # if segment is empty, xor is not defined
        if (not self) or (not other):
            raise ValueError(
                'The gap between a segment and an empty segment '
                'is not defined.')

        start = min(self.end, other.end)
        end = max(self.start, other.start)
        return Segment(start=start, end=end)

    def _str_helper(self, seconds: float) -> str:
        from datetime import timedelta
        negative = seconds < 0
        seconds = abs(seconds)
        td = timedelta(seconds=seconds)
        seconds = td.seconds + 86400 * td.days
        microseconds = td.microseconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '%s%02d:%02d:%02d.%03d' % (
            '-' if negative else ' ', hours, minutes,
            seconds, microseconds / 1000)

    def __str__(self):
        """Human-readable representation

        >>> print(Segment(1337, 1337 + 0.42))
        [ 00:22:17.000 -->  00:22:17.420]

        Note
        ----
        Empty segments are printed as "[]"
        """
        if self:
            return '[%s --> %s]' % (self._str_helper(self.start),
                                    self._str_helper(self.end))
        return '[]'

    def __repr__(self):
        """Computer-readable representation

        >>> Segment(1337, 1337 + 0.42)
        <Segment(1337, 1337.42)>
        """
        return '<Segment(%g, %g)>' % (self.start, self.end)

    def for_json(self):
        """Serialization

        See also
        --------
        :mod:`pyannote.core.json`
        """
        return {'start': self.start, 'end': self.end}

    @classmethod
    def from_json(cls, data):
        """Deserialization

        See also
        --------
        :mod:`pyannote.core.json`
        """
        return cls(start=data['start'], end=data['end'])

Segment.set_precision(2)