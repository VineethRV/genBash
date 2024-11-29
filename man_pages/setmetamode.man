SETMETAMODE(1)              General Commands Manual             SETMETAMODE(1)

NAME
       setmetamode - define the keyboard meta key handling

SYNOPSIS
       setmetamode [ options ] [ argument ]

DESCRIPTION
       Without  argument,  setmetamode prints the current Meta key mode.  With
       argument, it sets the Meta key mode as indicated.  The  setting  before
       and after the change are reported.

       The  Meta key mode is specific for each VT (and the VT corresponding to
       stdin is used).  One might use setmetamode in  /etc/rc  to  define  the
       initial state of the Meta key mode, e.g. by

            INITTY=/dev/tty[1-8]
            for tty in $INITTY; do
                 setmetamode escprefix < $tty
            done

ARGUMENTS
       esc, prefix, escprefix
              The Meta key sends an Escape prefix.

       meta, bit, metabit
              The Meta key sets the high order bit of the character.

OPTIONS
       -C, --console=DEV
              the console device to be used;

       -V, --version
              print version number;

       -h, --help
              print this usage message.

SEE ALSO
       loadkeys(1), kbdinfo(1)

kbd                               30 Jan 1994                   SETMETAMODE(1)
